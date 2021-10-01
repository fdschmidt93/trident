from types import MethodType
from typing import Any, Callable, Optional, Union

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.parsing import AttributeDict
from torch import nn
from transformers.file_utils import ModelOutput
from transformers.tokenization_utils_base import BatchEncoding

from src.models.mixin.eval import EvalMixin
from src.models.mixin.optimizer import OptimizerMixin


class Model(OptimizerMixin, EvalMixin, LightningModule):
    hparams: AttributeDict

    def __init__(
        self,
        model: DictConfig,
        optimizer: DictConfig,
        scheduler: Optional[DictConfig],
        evaluation: Optional[DictConfig] = None,
        overrides: Optional[DictConfig] = None,
        **kwargs: Any,
    ):
        self.save_hyperparameters()
        super().__init__()
        self.model: nn.Module = hydra.utils.instantiate(self.hparams.model)
        # set up a config
        # that forward:
        #   __target__: src.utils.my_function

        # for comfort:
        self.overrides = hydra.utils.instantiate(self.hparams.overrides)
        if self.overrides is not None:
            for key, value in self.overrides.items():
                setattr(self, key, MethodType(value, self))
        # TODO(fdschmidt93): define mixin on-the-fly

    def forward(self, batch: BatchEncoding) -> ModelOutput:
        """Plain forward pass of your model for which the batch is unpacked."""
        return self.model(**batch)

    def training_step(
        self, batch: BatchEncoding, batch_idx: int
    ) -> Union[dict[str, Any], ModelOutput]:
        """Training step of your model which takes a forward pass.

        Notes:
            If you want to extend `training_step`, add a `on_train_batch_end` method via overrides.
            See: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#on-train-batch-end

        Args:
            batch: typically comprising input_ids, attention_mask, and position_ids
            batch_idx: variable used internally by pytorch-lightning

        Returns:
            Union[dict[str, Any], ModelOutput]: model output that must have 'loss' as attr or key
        """
        outputs = self.model(**batch)
        self.log("train/loss", outputs.loss)
        return outputs
