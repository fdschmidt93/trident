from typing import Optional

import hydra
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.parsing import AttributeDict

from src.utils import utils

log = utils.get_logger(__name__)


class OptimizerMixin:
    hparams: AttributeDict

    def __init__(self) -> None:
        super().__init__()

    @property
    def num_training_steps(self):
        accumulate_grad_batches = getattr(self.trainer, "accumulate_grad_batches", 1)
        if self.trainer.datamodule is not None:
            num_training_samples = len(self.trainer.datamodule.train_dataloader())
        else:
            num_training_samples = len(self.train_dataloader())
        return (
            num_training_samples
            * self.trainer.max_epochs
            // max(1, self.trainer.num_gpus)
            // accumulate_grad_batches
        )

    def configure_scheduler(self, optimizer):
        if hasattr(self.hparams.scheduler, "num_warmup_steps") and isinstance(
            self.hparams.scheduler.num_warmup_steps, float
        ):
            self.hparams.scheduler.num_warmup_steps *= self.num_training_steps
        self.hparams.scheduler.num_training_steps = self.num_training_steps
        log.info(
            f"Warm up for {self.hparams.scheduler.num_warmup_steps} of {self.num_training_steps}"
        )
        scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer,)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        if hasattr(self.hparams.optimizer, "weight_decay"):
            param_optimizer = list(self.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.hparams.optimizer.weight_decay,
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            parameters = self.parameters()

        optimizer = hydra.utils.instantiate(self.hparams.optimizer, parameters)
        if hasattr(self.hparams, "scheduler"):
            scheduler = self.configure_scheduler(optimizer)
            return [optimizer], [scheduler]
        return [optimizer]
