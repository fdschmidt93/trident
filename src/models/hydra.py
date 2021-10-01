from typing import Optional, Union

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from src.models.mixin.eval import EvalMixin
from src.models.mixin.optimizer import OptimizerMixin
from src.models.modules.utils import get_metrics


class HydraModel(EvalMixin, OptimizerMixin, LightningModule):
    def __init__(
        self,
        encoder: DictConfig,
        head: DictConfig,
        optimizer: DictConfig,
        pooler: Optional[DictConfig] = None,
        metrics: dict = {"Accuracy": "preds"},
        *args,
        **kwargs,
    ):
        super().__init__()

        self.encoder = hydra.utils.instantiate(self.hparams.encoder)
        self.pooler = (
            hydra.utils.instantiate(self.hparams.pooler)
            if pooler is not None
            else lambda x: x
        )
        self.head = hydra.utils.instantiate(self.hparams.head)

        self.val_metrics = get_metrics(metrics)
        self.test_metrics = get_metrics(metrics)
        self.train()

    def embed(self, batch: dict) -> torch.Tensor:
        return self.encoder(**batch)

    def forward(self, batch: dict) -> torch.Tensor:
        hidden_states = self.embed(batch)
        embeds = self.pooler(hidden_states, **batch)
        outputs = self.head(embeds, batch["labels"])
        return outputs

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self(**batch)
        loss: torch.Tensor = outputs.loss
        self.log("train_loss", loss)
        return loss
