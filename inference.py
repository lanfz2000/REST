#!/usr/bin/env python3
"""Extract REST features or apply Complementary Prediction Fusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rest.model import HOptimus1Encoder, image_transform


class ImageDataset(Dataset):
    def __init__(self, source: Path) -> None:
        self.source = source
        self.transform = image_transform()
        self.handle: h5py.File | None = None
        if source.suffix == ".h5":
            with h5py.File(source, "r") as handle:
                self.image_key = next(key for key in ("img", "imgs", "images") if key in handle)
                self.length = len(handle[self.image_key])
            self.paths: list[Path] = []
        else:
            extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
            self.paths = sorted(path for path in source.rglob("*") if path.suffix.lower() in extensions)
            self.length = len(self.paths)
            if not self.paths:
                raise FileNotFoundError(f"No images found under {source}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.paths:
            image = Image.open(self.paths[index]).convert("RGB")
        else:
            if self.handle is None:
                self.handle = h5py.File(self.source, "r")
            image = Image.fromarray(self.handle[self.image_key][index]).convert("RGB")
        return self.transform(image)


def extract(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = HOptimus1Encoder(args.hoptimus1)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        encoder.load_state_dict(state, strict=True)
    encoder.to(device).eval()
    dataset = ImageDataset(args.input)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    features = []
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        for images in tqdm(loader):
            features.append(encoder(images.to(device, non_blocking=True)).float().cpu())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, torch.cat(features).numpy())


def fuse(args: argparse.Namespace) -> None:
    raw = np.load(args.raw)
    adapted = np.load(args.adapted)
    if raw.shape != adapted.shape:
        raise ValueError(f"Prediction shapes differ: {raw.shape} and {adapted.shape}")
    fused = raw + args.alpha * (adapted - raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, fused.astype(np.float32, copy=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--input", type=Path, required=True, help="Image directory or patch H5")
    extract_parser.add_argument("--checkpoint", type=Path, help="REST encoder.pt; omit for the original encoder")
    extract_parser.add_argument("--hoptimus1", default="hf-hub:bioptimus/H-optimus-1")
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--batch-size", type=int, default=64)
    extract_parser.add_argument("--num-workers", type=int, default=4)
    extract_parser.set_defaults(function=extract)
    fuse_parser = subparsers.add_parser("fuse")
    fuse_parser.add_argument("--raw", type=Path, required=True)
    fuse_parser.add_argument("--adapted", type=Path, required=True)
    fuse_parser.add_argument("--alpha", type=float, default=0.75)
    fuse_parser.add_argument("--output", type=Path, required=True)
    fuse_parser.set_defaults(function=fuse)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
