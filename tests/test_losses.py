import torch

from rest.losses import bipr_loss, spot_reliability, srca_loss


def test_spot_reliability_is_normalized_and_nonuniform() -> None:
    expression = torch.tensor([[1.0, 1.0, 1.0], [0.0, 4.0, 8.0]])
    observed = torch.ones_like(expression)
    membership = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.bool)
    reliability = spot_reliability(expression, observed, membership)
    assert torch.isclose(reliability.mean(), torch.tensor(1.0))
    assert reliability[1] > reliability[0]


def test_srca_prefers_aligned_pairs() -> None:
    embeddings = torch.eye(3)
    loss, accuracy = srca_loss(
        embeddings, embeddings, torch.tensor([0.5, 1.0, 1.5]), torch.tensor(2.0)
    )
    assert loss.item() < 0.01
    assert accuracy.item() == 1.0


def test_bipr_zero_for_exact_reconstruction() -> None:
    expression = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    observed = torch.ones_like(expression)
    indices = [torch.tensor([0, 1]), torch.tensor([1, 2])]
    predictions = [expression[:, [0, 1]], expression[:, [1, 2]]]
    loss, pearson = bipr_loss(
        predictions,
        expression,
        observed,
        indices,
        [torch.ones(2), torch.ones(2)],
        [0, 1],
    )
    assert loss.item() == 0.0
    assert torch.isclose(pearson, torch.tensor(1.0))
