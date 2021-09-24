import importlib
from src.models.huggingface import HFModel
from typing import Any, Union

import hydra
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.core.lightning import LightningModule

# cfg = OmegaConf.load("./configs/mixin/base.yaml")
# mixins = [getattr(importlib.import_module(k), v) for k, v in cfg.items()]

Model = None
def compose_model(cfg: Union[dict, OmegaConf, DictConfig]) -> LightningModule:
    mixins = [getattr(importlib.import_module(k), v) for k, v in cfg.items()][1:]

    class LightningModel(*mixins, HFModel):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)

    global Model    
    Model = LightningModel
