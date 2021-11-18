from typing import Union

import hydra
from pytorch_lightning.utilities.parsing import AttributeDict

from trident.utilities.logging import get_logger

log = get_logger(__name__)

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class OptimizerMixin:
    """Mixin for base model to define configuration of optimizer and scheduler.

    The OptimizerMixin provides functionality to:
        * compute the number of training steps (:obj:`OptimizerMixin.num_training_steps`)
        * configure the optimizer(s) (:obj:`OptimizerMixin.configure_optimizers`)
        * configure the scheduler (:obj:`OptimizerMixin.configure_scheduler`)

    Examples:
        * Optimizer: :repo:`AdamW <configs/optimizer/adamw.yaml>`
        * Scheduler: :repo:`Linear Warm-Up <configs/scheduler/linear_warm_up.yaml>`

    """

    hparams: AttributeDict

    @property
    def num_training_steps(self):
        """Computes the number of training steps per device, accounting for gradient accumulation."""
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

    def configure_scheduler(
        self, optimizer: Optimizer
    ) -> dict[str, Union[str, int, LambdaLR]]:
        """Configures the LR scheduler for the optimizer.

        The instantiation of the scheduler takes the optimizer as the first positional argument.

        .. code-block:: python

            # hparams.scheduler: passed config
            scheduler: LambdaLR = hydra.utils.instantiate(self.hparams.scheduler, optimizer,)


        Note that the below values are hard-coded for the time being:
            * interval: step
            * frequency: 1

        Args:
            optimizer: pytorch optimizer

        Returns:
            dict[str, Union[str, int, LambdaLR]: scheduler in pytorch-lightning format
        """
        if hasattr(self.hparams.scheduler, "num_warmup_steps") and isinstance(
            self.hparams.scheduler.num_warmup_steps, float
        ):
            self.hparams.scheduler.num_warmup_steps *= self.num_training_steps
        self.hparams.scheduler.num_training_steps = self.num_training_steps
        log.info(
            f"Warm up for {self.hparams.scheduler.num_warmup_steps} of {self.num_training_steps}"
        )
        scheduler: LambdaLR = hydra.utils.instantiate(
            self.hparams.scheduler,
            optimizer,
        )
        # TODO(fdschmidt93): more flexible LR schedules?
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepares optimizer and scheduler."""
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
