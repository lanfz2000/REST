#!/usr/bin/env python3
"""Train REST with paired H&E patches and spatial transcriptomics."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from rest.data import MixedSampleBatchSampler, PairedHESTDataset, RankSplitBatchSampler
from rest.losses import bipr_loss, infonce_loss, spot_reliability, srca_loss
from rest.model import HOptimus1Encoder, RESTModel, add_qv_lora, merged_encoder_state_dict


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--patch-root", required=True)
    parser.add_argument("--st-root", required=True)
    parser.add_argument("--program-prefix", required=True)
    parser.add_argument(
        "--ensembl-map",
        default=str(Path(__file__).with_name("resources") / "ensembl_to_symbol.tsv"),
    )
    parser.add_argument("--hoptimus1", default="hf-hub:bioptimus/H-optimus-1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256, help="Global batch size")
    parser.add_argument("--samples-per-batch", type=int, default=32)
    parser.add_argument("--max-spots-per-sample", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--encoder-lr", type=float, default=5e-4)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--cell-weight", type=float, default=0.25)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-val-rows", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def owner(model: torch.nn.Module) -> RESTModel:
    return model.module if isinstance(model, DistributedDataParallel) else model


def gather(values: torch.Tensor, gradients: bool = True) -> torch.Tensor:
    if not dist.is_initialized():
        return values
    if gradients:
        return torch.cat(dist_nn.all_gather(values), dim=0)
    gathered = [torch.empty_like(values) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, values)
    return torch.cat(gathered, dim=0)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cell_weight: float,
    reconstruction_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    module = owner(model)
    totals = {"loss": 0.0, "srca": 0.0, "bipr": 0.0, "pearson": 0.0, "top1": 0.0}
    count = 0
    progress = tqdm(loader, disable=dist.is_initialized() and dist.get_rank() != 0)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        expression = batch["expression"].to(device, non_blocking=True)
        observed = batch["observed"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            output = model(images, expression, observed)
            image_embedding = gather(output["image_embedding"])
            st_embedding = gather(output["st_embedding"])
            if training:
                reliability = spot_reliability(expression, observed, module.membership)
                reliability = gather(reliability, gradients=False)
                alignment, top1 = srca_loss(
                    image_embedding, st_embedding, reliability, module.logit_scale
                )
            else:
                alignment, top1 = infonce_loss(
                    image_embedding, st_embedding, module.logit_scale
                )
            reconstruction, pearson = bipr_loss(
                output["program_predictions"],
                expression,
                observed,
                module.program_indices(),
                module.overlap_weights(),
                module.source_groups,
                cell_weight,
            )
            loss = alignment + reconstruction_weight * reconstruction
        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            module.logit_scale.data.clamp_(max=math.log(100.0))

        batch_size = images.shape[0]
        count += batch_size
        values = {
            "loss": loss,
            "srca": alignment,
            "bipr": reconstruction,
            "pearson": pearson,
            "top1": top1,
        }
        for key, value in values.items():
            totals[key] += float(value.detach()) * batch_size

    if dist.is_initialized():
        packed = torch.tensor(
            [*totals.values(), count], dtype=torch.float64, device=device
        )
        dist.all_reduce(packed)
        count = int(packed[-1])
        totals = {key: float(packed[index]) for index, key in enumerate(totals)}
    return {key: value / max(1, count) for key, value in totals.items()}


def save_checkpoint(path: Path, model: torch.nn.Module, metadata: dict[str, object]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    module = owner(model)
    torch.save(merged_encoder_state_dict(module.encoder), path / "encoder.pt")
    heads = {
        name: value.detach().cpu()
        for name, value in module.state_dict().items()
        if not name.startswith("encoder.")
    }
    torch.save(heads, path / "rest_heads.pt")
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = arguments()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed + rank)
    torch.set_float32_matmul_precision("high")

    prefix = Path(args.program_prefix)
    metadata = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    membership = torch.from_numpy(
        sp.load_npz(prefix.with_suffix(".npz")).toarray().astype(np.bool_)
    )
    if list(membership.shape) != metadata["shape"]:
        raise ValueError("Program matrix and metadata shapes differ")

    train_set = PairedHESTDataset(
        args.index,
        args.patch_root,
        args.st_root,
        metadata["genes"],
        "train",
        ensembl_map=args.ensembl_map,
        max_rows=args.max_train_rows,
        seed=args.seed,
    )
    val_set = PairedHESTDataset(
        args.index,
        args.patch_root,
        args.st_root,
        metadata["genes"],
        "val",
        ensembl_map=args.ensembl_map,
        max_rows=args.max_val_rows,
        seed=args.seed,
    )
    global_sampler = MixedSampleBatchSampler(
        train_set.frame,
        args.batch_size,
        args.samples_per_batch,
        args.max_spots_per_sample,
        args.seed,
    )
    train_sampler = (
        RankSplitBatchSampler(global_sampler, rank, world_size)
        if distributed
        else global_sampler
    )
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None
    workers = max(0, args.num_workers // world_size)
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(2, args.batch_size // world_size),
        sampler=val_sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    encoder = HOptimus1Encoder(args.hoptimus1)
    add_qv_lora(encoder)
    model = RESTModel(
        encoder,
        membership,
        metadata["pathway_sources"],
    ).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

    module = owner(model)
    encoder_parameters = [
        parameter
        for name, parameter in module.named_parameters()
        if name.startswith("encoder.") and parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for name, parameter in module.named_parameters()
        if not name.startswith("encoder.") and parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup = min(args.warmup_steps, max(0, total_steps - 1))

    def learning_rate_scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    best, stale = -math.inf, 0
    checkpoint_metadata = {
        "method": "REST",
        "backbone": "H-optimus-1",
        "programs": int(membership.shape[0]),
        "genes": int(membership.shape[1]),
        "cell_weight": args.cell_weight,
        "reconstruction_weight": args.reconstruction_weight,
    }
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args.cell_weight,
            args.reconstruction_weight,
            optimizer,
            scaler,
            scheduler,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            args.cell_weight,
            args.reconstruction_weight,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)
            (args.output / "history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
            save_checkpoint(args.output / "last", model, {**checkpoint_metadata, **row})
        if val_metrics["pearson"] > best:
            best, stale = val_metrics["pearson"], 0
            if rank == 0:
                save_checkpoint(args.output / "best", model, {**checkpoint_metadata, **row})
        else:
            stale += 1
        if args.early_stop_patience and stale >= args.early_stop_patience:
            break

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
