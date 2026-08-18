"""REST model components."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms


HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)


def image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(HOPTIMUS_MEAN, HOPTIMUS_STD),
        ]
    )


class HOptimus1Encoder(nn.Module):
    """H-optimus-1 wrapper exposing pooled and patch-token features."""

    feature_dim = 1536
    token_dim = 1536

    def __init__(self, model_path: str = "hf-hub:bioptimus/H-optimus-1") -> None:
        super().__init__()
        import timm

        source = f"local-dir:{model_path}" if Path(model_path).is_dir() else model_path
        self.model = timm.create_model(
            source,
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )

    def forward_tokens(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.model.forward_features(images)
        pooled = self.model.forward_head(tokens, pre_logits=True)
        prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))
        return pooled, tokens[:, prefix_tokens:]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(images)[0]


class QVLoRALinear(nn.Module):
    """LoRA updates for the query and value slices of a fused qkv layer."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if base.out_features != 3 * base.in_features:
            raise ValueError("Expected a fused qkv projection")
        self.base = base
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        width = base.in_features
        self.q_a = nn.Parameter(torch.empty(rank, width))
        self.q_b = nn.Parameter(torch.zeros(width, rank))
        self.v_a = nn.Parameter(torch.empty(rank, width))
        self.v_b = nn.Parameter(torch.zeros(width, rank))
        nn.init.kaiming_uniform_(self.q_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_a, a=math.sqrt(5))
        for parameter in base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dropped = self.dropout(inputs)
        query = F.linear(F.linear(dropped, self.q_a), self.q_b)
        value = F.linear(F.linear(dropped, self.v_a), self.v_b)
        update = torch.cat([query, torch.zeros_like(query), value], dim=-1)
        return self.base(inputs) + self.scaling * update

    def merged_weight(self) -> torch.Tensor:
        query = self.q_b.detach() @ self.q_a.detach()
        value = self.v_b.detach() @ self.v_a.detach()
        key = torch.zeros_like(query)
        return self.base.weight.detach() + self.scaling * torch.cat([query, key, value])


def add_qv_lora(
    encoder: HOptimus1Encoder,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.1,
) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    for block in encoder.model.blocks:
        block.attn.qkv = QVLoRALinear(block.attn.qkv, rank, alpha, dropout)


def merged_encoder_state_dict(encoder: HOptimus1Encoder) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    lora_names = {
        name for name, module in encoder.named_modules() if isinstance(module, QVLoRALinear)
    }
    for name, module in encoder.named_modules():
        if isinstance(module, QVLoRALinear):
            merged[f"{name}.weight"] = module.merged_weight().cpu()
            if module.base.bias is not None:
                merged[f"{name}.bias"] = module.base.bias.detach().cpu()
    for name, value in encoder.state_dict().items():
        if not any(name.startswith(f"{prefix}.") for prefix in lora_names):
            merged[name] = value.detach().cpu()
    return merged


class ProgramEncoder(nn.Module):
    def __init__(
        self,
        membership: torch.Tensor,
        hidden_dim: int = 256,
        embed_dim: int = 256,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoders = nn.ModuleList()
        for index, row in enumerate(membership):
            indices = torch.where(row.bool())[0]
            self.register_buffer(f"indices_{index}", indices, persistent=False)
            self.encoders.append(nn.Sequential(nn.Linear(len(indices), hidden_dim), nn.SELU()))
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            heads,
            hidden_dim * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.projector = nn.Linear(hidden_dim, embed_dim)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self, expression: torch.Tensor, observed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expression = expression * observed
        tokens = [
            encoder(expression.index_select(1, getattr(self, f"indices_{index}")))
            for index, encoder in enumerate(self.encoders)
        ]
        tokens = torch.stack(tokens, dim=1)
        cls = self.cls_token.expand(expression.shape[0], -1, -1)
        encoded = self.transformer(torch.cat([cls, tokens], dim=1))
        return F.normalize(self.projector(encoded[:, 0]), dim=1), encoded[:, 1:]


class RESTModel(nn.Module):
    """SRCA and BIPR on top of a pathology foundation encoder."""

    def __init__(
        self,
        encoder: HOptimus1Encoder,
        membership: torch.Tensor,
        program_sources: list[str],
        hidden_dim: int = 256,
        embed_dim: int = 256,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if membership.shape[0] != len(program_sources):
            raise ValueError("program_sources must align with membership rows")
        self.encoder = encoder
        self.register_buffer("membership", membership.bool(), persistent=True)
        self.image_projector = nn.Linear(encoder.feature_dim, embed_dim)
        self.visual_projector = nn.Linear(encoder.token_dim, hidden_dim)
        self.program_encoder = ProgramEncoder(
            membership, hidden_dim, embed_dim, encoder_layers, heads, dropout
        )
        decoder_layer = nn.TransformerDecoderLayer(
            hidden_dim,
            heads,
            hidden_dim * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, decoder_layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.program_queries = nn.Parameter(
            torch.empty(1, membership.shape[0], hidden_dim)
        )
        self.reconstruction_heads = nn.ModuleList()
        self.source_groups = [
            1 if source == "PanglaoDBCanonical" else 0 for source in program_sources
        ]

        group_counts = []
        for group in range(2):
            rows = [i for i, value in enumerate(self.source_groups) if value == group]
            if rows:
                group_counts.append(membership[rows].float().sum(0).clamp_min(1.0))
            else:
                group_counts.append(torch.ones(membership.shape[1]))
        for index, row in enumerate(membership):
            indices = torch.where(row.bool())[0]
            weights = group_counts[self.source_groups[index]].index_select(0, indices).reciprocal()
            self.register_buffer(f"program_indices_{index}", indices, persistent=False)
            self.register_buffer(f"overlap_weights_{index}", weights, persistent=False)
            self.reconstruction_heads.append(nn.Linear(hidden_dim, len(indices)))

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        nn.init.normal_(self.program_queries, std=0.02)

    def program_indices(self) -> list[torch.Tensor]:
        return [
            getattr(self, f"program_indices_{index}")
            for index in range(len(self.reconstruction_heads))
        ]

    def overlap_weights(self) -> list[torch.Tensor]:
        return [
            getattr(self, f"overlap_weights_{index}")
            for index in range(len(self.reconstruction_heads))
        ]

    def forward(
        self,
        images: torch.Tensor,
        expression: torch.Tensor,
        observed: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        pooled, visual_tokens = self.encoder.forward_tokens(images)
        st_embedding, _ = self.program_encoder(expression, observed)
        memory = self.visual_projector(visual_tokens)
        queries = self.program_queries.expand(images.shape[0], -1, -1)
        decoded = self.decoder(queries, memory)
        predictions = [
            head(decoded[:, index])
            for index, head in enumerate(self.reconstruction_heads)
        ]
        return {
            "image_embedding": F.normalize(self.image_projector(pooled.float()), dim=1),
            "st_embedding": st_embedding,
            "program_predictions": predictions,
        }
