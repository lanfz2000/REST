"""Local HEST patch and expression loading."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .model import image_transform


def decode_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).reshape(-1)
    ]


def read_ensembl_map(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    frame = pd.read_csv(path, sep="\t", dtype=str)
    return {
        ensembl.split(".")[0].upper(): symbol.upper()
        for ensembl, symbol in zip(frame["ensembl_id"], frame["gene_symbol"])
        if pd.notna(ensembl) and pd.notna(symbol)
    }


def _gene_symbols(handle: h5py.File, ensembl_map: dict[str, str]) -> list[str]:
    var = handle["var"]
    for key in ("feature_name", "gene_symbol", "gene_name", "_index"):
        if key in var and isinstance(var[key], h5py.Dataset):
            identifiers = [value.split(".")[0].upper() for value in decode_strings(var[key][()])]
            return [ensembl_map.get(identifier, identifier) for identifier in identifiers]
    raise KeyError("No gene symbols found under h5ad/var")


def _sparse_matrix(group: h5py.Group) -> sp.spmatrix:
    encoding = group.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    values = (
        np.asarray(group["data"], dtype=np.float32),
        np.asarray(group["indices"], dtype=np.int32),
        np.asarray(group["indptr"], dtype=np.int64),
    )
    shape = tuple(int(value) for value in group.attrs["shape"])
    if encoding == "csr_matrix":
        return sp.csr_matrix(values, shape=shape)
    if encoding == "csc_matrix":
        return sp.csc_matrix(values, shape=shape).tocsr()
    raise ValueError(f"Unsupported h5ad X encoding: {encoding}")


class PairedHESTDataset(Dataset):
    def __init__(
        self,
        index_csv: str | Path,
        patch_root: str | Path,
        st_root: str | Path,
        genes: list[str],
        split: str,
        ensembl_map: str | Path | None = None,
        library_size: float = 10_000.0,
        max_rows: int | None = None,
        seed: int = 42,
    ) -> None:
        columns = {
            "sample_id",
            "spot_id",
            "split",
            "patch_h5_index",
            "expression_row_index",
        }
        frame = pd.read_csv(index_csv, low_memory=False)
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"Index is missing columns: {sorted(missing)}")
        frame = frame[frame["split"].astype(str) == split].copy()
        if max_rows and len(frame) > max_rows:
            frame = frame.sample(max_rows, random_state=seed)
        self.frame = frame.reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No rows found for split={split!r}")
        self.patch_root = Path(patch_root)
        self.st_root = Path(st_root)
        self.genes = [gene.upper() for gene in genes]
        self.gene_to_index = {gene: index for index, gene in enumerate(self.genes)}
        self.ensembl_map = read_ensembl_map(ensembl_map)
        self.library_size = library_size
        self.transform = image_transform()
        sample_ids = sorted(self.frame["sample_id"].astype(str).unique())
        self.sample_to_index = {sample: index for index, sample in enumerate(sample_ids)}
        self.patch_handles: dict[str, h5py.File] = {}
        self.st_handles: dict[str, h5py.File] = {}
        self.st_matrices: dict[str, sp.spmatrix | h5py.Dataset | h5py.Group] = {}
        self.local_to_union: dict[str, np.ndarray] = {}
        self.observed: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.frame)

    @staticmethod
    def _evict(handles: dict[str, h5py.File], limit: int = 8) -> None:
        while len(handles) >= limit:
            _, handle = handles.popitem()
            handle.close()

    def _open_patch(self, sample: str) -> h5py.File:
        if sample not in self.patch_handles:
            self._evict(self.patch_handles)
            self.patch_handles[sample] = h5py.File(self.patch_root / f"{sample}.h5", "r")
        return self.patch_handles[sample]

    def _open_st(self, sample: str) -> h5py.File:
        if sample in self.st_handles:
            return self.st_handles[sample]
        if len(self.st_handles) >= 8:
            evicted, handle = self.st_handles.popitem()
            handle.close()
            self.st_matrices.pop(evicted, None)
            self.local_to_union.pop(evicted, None)
            self.observed.pop(evicted, None)
        handle = h5py.File(self.st_root / f"{sample}.h5ad", "r")
        symbols = _gene_symbols(handle, self.ensembl_map)
        mapping = np.full(len(symbols), -1, dtype=np.int32)
        observed = np.zeros(len(self.genes), dtype=np.float32)
        for local_index, symbol in enumerate(symbols):
            union_index = self.gene_to_index.get(symbol)
            if union_index is not None:
                mapping[local_index] = union_index
                observed[union_index] = 1.0
        x = handle["X"]
        if isinstance(x, h5py.Dataset):
            matrix: sp.spmatrix | h5py.Dataset | h5py.Group = x
        else:
            encoding = x.attrs.get("encoding-type", "")
            if isinstance(encoding, bytes):
                encoding = encoding.decode()
            matrix = x if encoding == "csr_matrix" else _sparse_matrix(x)
        self.st_matrices[sample] = matrix
        self.local_to_union[sample] = mapping
        self.observed[sample] = observed
        self.st_handles[sample] = handle
        return handle

    def _read_expression(self, sample: str, row_index: int) -> tuple[np.ndarray, np.ndarray]:
        self._open_st(sample)
        matrix = self.st_matrices[sample]
        if isinstance(matrix, h5py.Dataset):
            row = np.asarray(matrix[row_index], dtype=np.float32).reshape(-1)
            local_indices = np.arange(len(row))
            values = row
        elif isinstance(matrix, h5py.Group):
            indptr = matrix["indptr"]
            start, end = int(indptr[row_index]), int(indptr[row_index + 1])
            local_indices = np.asarray(matrix["indices"][start:end], dtype=np.int64)
            values = np.asarray(matrix["data"][start:end], dtype=np.float32)
        else:
            row = matrix.getrow(row_index)
            local_indices = row.indices
            values = row.data.astype(np.float32, copy=False)
        output = np.zeros(len(self.genes), dtype=np.float32)
        union_indices = self.local_to_union[sample][local_indices]
        valid = union_indices >= 0
        np.add.at(output, union_indices[valid], values[valid])
        total = float(values.sum())
        if total > 0:
            output = np.log1p(output * (self.library_size / total)).astype(np.float32)
        return output, self.observed[sample]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        sample = str(row["sample_id"])
        patch_handle = self._open_patch(sample)
        image_key = next(key for key in ("img", "imgs", "images") if key in patch_handle)
        image = Image.fromarray(
            patch_handle[image_key][int(row["patch_h5_index"])]
        ).convert("RGB")
        expression, observed = self._read_expression(
            sample, int(row["expression_row_index"])
        )
        return {
            "image": self.transform(image),
            "expression": torch.from_numpy(expression),
            "observed": torch.from_numpy(observed.copy()),
            "sample_index": torch.tensor(self.sample_to_index[sample]),
        }


class MixedSampleBatchSampler(Sampler[list[int]]):
    """Build each batch from several slides while limiting spots per slide."""

    def __init__(
        self,
        frame: pd.DataFrame,
        batch_size: int,
        samples_per_batch: int,
        max_spots_per_sample: int,
        seed: int,
    ) -> None:
        self.groups = {
            str(sample): indices.tolist()
            for sample, indices in frame.reset_index().groupby("sample_id")["index"]
        }
        self.samples_per_batch = samples_per_batch
        self.per_sample = max(1, batch_size // samples_per_batch)
        self.max_spots = max_spots_per_sample
        self.seed = seed
        self.epoch = 0
        chunks = sum(
            math.ceil(min(len(indices), max_spots_per_sample) / self.per_sample)
            for indices in self.groups.values()
        )
        self.length = math.ceil(chunks / samples_per_batch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        queues: dict[str, list[list[int]]] = {}
        for sample, source in self.groups.items():
            indices = source[:]
            rng.shuffle(indices)
            indices = indices[: self.max_spots]
            queues[sample] = [
                indices[start : start + self.per_sample]
                for start in range(0, len(indices), self.per_sample)
            ]
        samples = list(queues)
        rng.shuffle(samples)
        for start in range(0, len(samples), self.samples_per_batch):
            cohort = samples[start : start + self.samples_per_batch]
            while any(queues[sample] for sample in cohort):
                batch = []
                for sample in cohort:
                    if queues[sample]:
                        batch.extend(queues[sample].pop())
                if len(batch) >= 2:
                    yield batch

    def __len__(self) -> int:
        return self.length


class RankSplitBatchSampler(Sampler[list[int]]):
    def __init__(self, sampler: Sampler[list[int]], rank: int, world_size: int) -> None:
        self.sampler = sampler
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self.sampler:
            usable = len(batch) - len(batch) % self.world_size
            if usable >= 2 * self.world_size:
                yield batch[self.rank:usable:self.world_size]

    def __len__(self) -> int:
        return len(self.sampler)
