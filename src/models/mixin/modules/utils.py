from typing import Any, Union

import hydra
from omegaconf.omegaconf import OmegaConf


class ApplyMap:
    def __init__(self, cfg: Union[dict, OmegaConf]) -> None:
        self.cfg = hydra.utils.instantiate(cfg)

    def __call__(self, inputs: Any) -> Any:
        for v in self.cfg.values():
            inputs = v(inputs)
        return inputs


def get_preds(outputs, *args, **kwargs):
    outputs.preds = outputs.logits.argmax(dim=-1)
    return outputs
