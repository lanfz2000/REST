"""Losses used by REST."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def spot_reliability(
    expression: torch.Tensor,
    observed: torch.Tensor,
    membership: torch.Tensor,
) -> torch.Tensor:
    """Estimate spot reliability from program coverage and activity dispersion."""
    membership = membership.to(expression.device, expression.dtype)
    masked_expression = expression.float() * observed.float()
    counts = observed.float() @ membership.T
    activities = (masked_expression @ membership.T) / counts.clamp_min(1.0)
    coverage = counts / membership.sum(dim=1).clamp_min(1.0)
    available = coverage.gt(0).float()
    available_count = available.sum(dim=1).clamp_min(1.0)
    mean = (activities * available).sum(dim=1) / available_count
    dispersion = (
        ((activities - mean[:, None]).square() * available).sum(dim=1)
        / available_count
    ).sqrt()
    mean_coverage = coverage.sum(dim=1) / available_count
    reliability = (dispersion * mean_coverage.sqrt()).clamp_min(1e-6).sqrt()
    return reliability / reliability.mean().clamp_min(1e-6)


def infonce_loss(
    image_embedding: torch.Tensor,
    st_embedding: torch.Tensor,
    logit_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logit_scale.exp().clamp(max=100.0) * image_embedding @ st_embedding.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = 0.5 * (
        F.cross_entropy(logits.float(), labels)
        + F.cross_entropy(logits.T.float(), labels)
    )
    accuracy = 0.5 * (
        (logits.argmax(1) == labels).float().mean()
        + (logits.argmax(0) == labels).float().mean()
    )
    return loss, accuracy


def srca_loss(
    image_embedding: torch.Tensor,
    st_embedding: torch.Tensor,
    reliability: torch.Tensor,
    logit_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric InfoNCE with unequal weights on matched H&E-ST pairs."""
    logits = logit_scale.exp().clamp(max=100.0) * image_embedding @ st_embedding.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    weight = reliability.float().clamp_min(1e-6)
    weight = weight / weight.mean().clamp_min(1e-6)
    row = F.cross_entropy(logits.float(), labels, reduction="none")
    column = F.cross_entropy(logits.T.float(), labels, reduction="none")
    loss = 0.5 * ((weight * row).sum() + (weight * column).sum()) / weight.sum()
    accuracy = 0.5 * (
        (logits.argmax(1) == labels).float().mean()
        + (logits.argmax(0) == labels).float().mean()
    )
    return loss, accuracy


def _masked_program_pearson(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    count = mask.sum(dim=0)
    valid = count >= 2
    if not bool(valid.any()):
        return None
    safe_count = count.clamp_min(1.0)
    pred_centered = (prediction - (prediction * mask).sum(0) / safe_count) * mask
    target_centered = (target - (target * mask).sum(0) / safe_count) * mask
    numerator = (pred_centered * target_centered).sum(0)
    denominator = torch.sqrt(
        pred_centered.square().sum(0) * target_centered.square().sum(0)
    ).clamp_min(1e-6)
    return (numerator / denominator)[valid].mean()


def bipr_loss(
    predictions: list[torch.Tensor],
    expression: torch.Tensor,
    observed: torch.Tensor,
    program_indices: list[torch.Tensor],
    overlap_weights: list[torch.Tensor],
    source_groups: list[int],
    cell_weight: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source-balanced member-gene reconstruction over biological programs."""
    numerators = [expression.new_zeros((), dtype=torch.float32) for _ in range(2)]
    denominators = [expression.new_zeros((), dtype=torch.float32) for _ in range(2)]
    correlations: list[torch.Tensor] = []
    for prediction, indices, weights, group in zip(
        predictions, program_indices, overlap_weights, source_groups
    ):
        target = expression.index_select(1, indices).float()
        mask = observed.index_select(1, indices).float()
        weighted_mask = mask * weights.to(mask.device).float().unsqueeze(0)
        numerators[group] += ((prediction.float() - target).square() * weighted_mask).sum()
        denominators[group] += weighted_mask.sum()
        correlation = _masked_program_pearson(prediction.float(), target, mask)
        if correlation is not None:
            correlations.append(correlation)

    losses = [n / d.clamp_min(1.0) for n, d in zip(numerators, denominators)]
    if denominators[1].item() == 0:
        loss = losses[0]
    elif denominators[0].item() == 0:
        loss = losses[1]
    else:
        loss = (1.0 - cell_weight) * losses[0] + cell_weight * losses[1]
    pearson = torch.stack(correlations).mean() if correlations else loss.new_zeros(())
    return loss, pearson
