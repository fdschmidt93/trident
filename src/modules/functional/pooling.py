from typing import Optional

import torch


def mean(
    input_embeds: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    """Mean pools input embeddings over attention mask.

    Args:
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`: input representations
        attention_mask (:obj:`torch.LongTensor` of shape :obj:`({0})`): indicates valid offets with 1

    Returns:
        torch.Tensor: [TODO:description]
    """
    return (input_embeds * attention_mask[:, :, None]).sum(1) / attention_mask.sum(
        -1, keepdim=True
    )


def cls(input_embeds: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return input_embeds[:, 0, :]
