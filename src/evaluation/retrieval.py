import torch


def mrr(preds: torch.Tensor) -> torch.Tensor:
    """Compute MRR from row-aligned matrices of square query-document pairs.

    `mrr` is primarily intended for BLI or sentence-translation retrieval.

    Args:
        preds: square matrix of ranking scores

    Returns:
        torch.Tensor: mean reciprocal rank
    """
    N = preds.shape[0]
    rankings = preds.argsort(dim=-1, descending=True) == torch.arange(N)[:, None]
    reciprocal_rank = 1 / (1 + rankings.float().argmax(dim=-1))
    return reciprocal_rank.mean()
