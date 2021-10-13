import functools
from dataclasses import dataclass
from typing import Any, Callable, List

from hydra.utils import get_method

from src.modules.functional.pooling import cls


@dataclass
class PartialWrapper:
    methods: List[Callable]

    def __call__(self, inputs) -> Any:
        for method in self.methods:
            inputs = method(inputs)
        return inputs


def partial(_partial_, *args, **kwargs):
    if isinstance(_partial_, list):
        methods = PartialWrapper([get_method(p) for p in _partial_])
        return methods
    return functools.partial(get_method(_partial_), *args, **kwargs)


def forward(self, batch):
    return self.model(**batch)


def get_cls(outputs, batch):
    outputs.cls = cls(outputs.last_hidden_state, batch["attention_mask"])
    return outputs


# list of dictionaries flattend to one dictionary
def prepare_retrieval_eval(outputs):
    num = outputs["cls"].shape[0]
    outputs["cls"] /= outputs["cls"].norm(2, dim=-1, keepdim=True)
    src_embeds = outputs["cls"][: num // 2]
    trg_embeds = outputs["cls"][num // 2 :]
    # (1000, 1000)
    preds = src_embeds @ trg_embeds.T
    # targets = (
    #     torch.zeros((num // 2, num // 2)).fill_diagonal_(1).long().to(src_embeds.device)
    # )
    return {
        "preds": preds,
        # "targets": targets,
    }
