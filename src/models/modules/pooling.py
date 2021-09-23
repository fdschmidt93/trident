import torch


def mean(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    return (hidden_states * attention_mask[:, :, None]).sum(1) / attention_mask.sum(
        -1, keepdim=True
    )


def cls(hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return hidden_states[:, 0, :]
