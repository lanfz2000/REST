#!/usr/bin/env python3
"""Match HEST patch barcodes to h5ad rows and create a REST index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from rest.data import decode_strings


def read_metadata(root: Path) -> pd.DataFrame:
    csv_path = root / "HEST_v1_3_0.csv"
    if csv_path.exists():
        frame = pd.read_csv(csv_path, low_memory=False)
        id_column = "id" if "id" in frame else "sample_id"
        frame[id_column] = frame[id_column].astype(str)
        return frame.set_index(id_column, drop=False)
    rows = []
    for path in (root / "metadata").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        row.setdefault("id", path.stem)
        rows.append(row)
    return pd.DataFrame(rows).set_index("id") if rows else pd.DataFrame()


def value_from(row: pd.Series | None, names: tuple[str, ...], default: str) -> str:
    if row is not None:
        for name in names:
            if name in row and pd.notna(row[name]):
                return str(row[name])
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hest-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--technologies", default="Visium,Visium HD")
    parser.add_argument("--species", default="Homo sapiens")
    parser.add_argument(
        "--split-csv",
        type=Path,
        help="Optional CSV with sample_id and split columns",
    )
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")

    patch_root = args.hest_root / "patches"
    st_root = args.hest_root / "st"
    samples = sorted(path.stem for path in patch_root.glob("*.h5") if (st_root / f"{path.stem}.h5ad").exists())
    if not samples:
        raise FileNotFoundError("No matching patches/*.h5 and st/*.h5ad files found")

    metadata = read_metadata(args.hest_root)
    technologies = {value.strip() for value in args.technologies.split(",") if value.strip()}
    if not metadata.empty:
        samples = [
            sample
            for sample in samples
            if sample in metadata.index
            and value_from(metadata.loc[sample], ("st_technology", "technology"), "unknown") in technologies
            and value_from(metadata.loc[sample], ("species",), "unknown") == args.species
        ]
    if not samples:
        raise ValueError("No samples remain after technology and species filtering")

    if args.split_csv:
        split_frame = pd.read_csv(args.split_csv, dtype=str)
        split_map = dict(zip(split_frame["sample_id"], split_frame["split"]))
        samples = [sample for sample in samples if sample in split_map]
    else:
        rng = np.random.default_rng(args.seed)
        shuffled = np.asarray(samples)
        rng.shuffle(shuffled)
        val_count = max(1, round(len(samples) * args.val_fraction))
        validation = set(shuffled[:val_count])
        split_map = {sample: "val" if sample in validation else "train" for sample in samples}
    if not samples or not {"train", "val"}.issubset(set(split_map.values())):
        raise ValueError("At least one train sample and one validation sample are required")

    rows: list[dict[str, object]] = []
    for sample in samples:
        with h5py.File(patch_root / f"{sample}.h5", "r") as patches, h5py.File(
            st_root / f"{sample}.h5ad", "r"
        ) as expression:
            barcode_key = next((key for key in ("barcode", "barcodes") if key in patches), None)
            if barcode_key is None:
                raise KeyError(f"No barcode dataset in {sample}.h5")
            patch_barcodes = decode_strings(patches[barcode_key][()])
            expression_barcodes = decode_strings(expression["obs"]["_index"][()])
            expression_rows = {barcode: index for index, barcode in enumerate(expression_barcodes)}
            coords = np.asarray(patches["coords"]) if "coords" in patches else None
            meta = metadata.loc[sample] if sample in metadata.index else None
            technology = value_from(meta, ("st_technology", "technology"), "unknown")
            species = value_from(meta, ("species",), "unknown")
            for patch_index, barcode in enumerate(patch_barcodes):
                expression_index = expression_rows.get(barcode)
                if expression_index is None:
                    continue
                x, y = (coords[patch_index] if coords is not None else (np.nan, np.nan))
                rows.append(
                    {
                        "sample_id": sample,
                        "spot_id": barcode,
                        "split": split_map[sample],
                        "patch_h5_index": patch_index,
                        "expression_row_index": expression_index,
                        "spatial_x": x,
                        "spatial_y": y,
                        "technology": technology,
                        "species": species,
                    }
                )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No matching patch-expression barcodes were found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"wrote {len(frame):,} pairs from {frame['sample_id'].nunique():,} samples to {args.output}")


if __name__ == "__main__":
    main()
