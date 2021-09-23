from pytorch_lightning import LightningModule
import hydra
from src.utils import utils

log = utils.get_logger(__name__)


class OptimizerMixin(LightningModule):
    def setup(self, stage):
        accumulate_grad_batches = self.trainer.accumulate_grad_batches
        accumulate_grad_batches = (
            accumulate_grad_batches if accumulate_grad_batches is not None else 1
        )
        if stage == "fit":

            if self.trainer.datamodule is not None:
                num_training_samples = len(self.trainer.datamodule.train_dataloader())
            else:
                num_training_samples = len(self.train_dataloader())
            self.total_steps = (
                (
                    num_training_samples
                    * self.trainer.max_epochs
                    // max(1, self.trainer.gpus)
                )
            ) // accumulate_grad_batches

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
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
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, optimizer_grouped_parameters
        )
        if hasattr(self.hparams.scheduler, "scheduler"):
            log.info(
                f"Warm up for {self.hparams.optimizer.warmup_steps * self.total_steps} of {self.total_steps}"
            )
            scheduler = hydra.utils.instantiate(
                self.hparams.optimizer.scheduler,
                optimizer,
                num_warmup_steps=self.scheduler.warmup_steps * self.total_steps,
                num_training_steps=self.total.steps,
            )

            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            return [optimizer], [scheduler]
        return [optimizer]
