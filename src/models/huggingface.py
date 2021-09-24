from typing import Any, Optional, Union

import hydra
import torch
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from torch import nn
from transformers.tokenization_utils_base import BatchEncoding

from src.models.mixin.eval import EvalMixin
from src.models.mixin.optimizer import OptimizerMixin


class HFModel(EvalMixin, OptimizerMixin, LightningModule):
    def __init__(
        self,
        model: Union[dict, DictConfig, OmegaConf],
        optimizer: Union[dict, DictConfig, OmegaConf],
        scheduler: Optional[Union[dict, DictConfig, OmegaConf]],
        metrics: dict = {"Accuracy": "preds"},
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model: nn.Module = hydra.utils.instantiate(self.hparams.model)

    def forward(self, batch: Union[dict, BatchEncoding]) -> dict:
        return self.model(**batch)

    def training_step(self, batch: BatchEncoding, batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        loss: torch.Tensor = outputs.loss
        self.log("train_loss", loss)
        return loss
