import torch


def mrr(preds: torch.Tensor) -> torch.Tensor:
    # (1000, 1000) -> diagonal sind meine true labels
    N = preds.shape[0]
    rankings = preds.argsort(dim=-1, descending=True) == torch.arange(N)[:, None]
    reciprocal_rank = 1 / (1 + rankings.float().argmax(dim=-1))
    return reciprocal_rank.mean()
