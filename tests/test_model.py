import torch
from torch import nn

from rest.losses import bipr_loss, spot_reliability, srca_loss
from rest.model import RESTModel


class TinyEncoder(nn.Module):
    feature_dim = 8
    token_dim = 8

    def forward_tokens(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = images.shape[0]
        pooled = torch.ones(batch, self.feature_dim)
        tokens = torch.ones(batch, 4, self.token_dim)
        return pooled, tokens


def test_rest_forward_shapes() -> None:
    membership = torch.tensor(
        [[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.bool
    )
    model = RESTModel(
        TinyEncoder(),
        membership,
        ["Hallmark", "KEGG", "PanglaoDBCanonical"],
        hidden_dim=8,
        embed_dim=4,
        encoder_layers=1,
        decoder_layers=1,
        heads=2,
        dropout=0,
    )
    output = model(torch.rand(2, 3, 8, 8), torch.rand(2, 4), torch.ones(2, 4))
    assert output["image_embedding"].shape == (2, 4)
    assert output["st_embedding"].shape == (2, 4)
    assert [value.shape for value in output["program_predictions"]] == [(2, 2)] * 3
    expression = torch.rand(2, 4)
    reliability = spot_reliability(expression, torch.ones(2, 4), membership)
    output = model(torch.rand(2, 3, 8, 8), expression, torch.ones(2, 4))
    alignment, _ = srca_loss(
        output["image_embedding"], output["st_embedding"], reliability, model.logit_scale
    )
    reconstruction, _ = bipr_loss(
        output["program_predictions"],
        expression,
        torch.ones(2, 4),
        model.program_indices(),
        model.overlap_weights(),
        model.source_groups,
    )
    (alignment + 0.1 * reconstruction).backward()
