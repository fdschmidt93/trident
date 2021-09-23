import importlib
from typing import Optional, Union

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding

from src.models.mixin.eval import EvalMixin
from src.models.mixin.optimizer import OptimizerMixin

from .modules.utils import get_metrics


class HFModel(EvalMixin, OptimizerMixin, LightningModule):
    def __init__(
        self,
        model: DictConfig,
        optimizer: DictConfig,
        metrics: dict = {"Accuracy": "preds"},
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = hydra.utils.instantiate(model)
        self.val_metrics = get_metrics(metrics)
        self.test_metrics = get_metrics(metrics)

    def forward(self, batch: BatchEncoding) -> dict:
        return self.model(**batch)

    def training_step(self, batch: BatchEncoding, batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        loss: torch.Tensor = outputs.loss
        self.log("train_loss", loss)
        return loss
