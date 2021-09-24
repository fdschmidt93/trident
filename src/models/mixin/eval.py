from typing import Any, Optional, Union

import hydra
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from transformers.tokenization_utils_base import BatchEncoding

OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)


class EvalMixin(LightningModule):
    def __init__(
        self,
        evaluation: Optional[dict, DictConfig, OmegaConf] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.evaluation = hydra.utils.instantiate(evaluation)
        if self.evaluation is not None:
            for transform in ["batch", "outputs"]:
                fn = getattr(self.evaluation.apply, transform)
                setattr(
                    self.evaluation.apply,
                    transform,
                    fn if callable(fn) else lambda input: input,
                )
            for metric in ["val_metrics", "test_metrics"]:
                setattr(
                    self,
                    metric,
                    getattr(self.evaluation.metrics, metric, self.evaluation.metrics),
                )

    def eval_step(self, batch: Union[dict, BatchEncoding]) -> dict:
        return self.evaluation.apply.outputs(self(batch))

    def validation_step(self, batch, batch_idx):
        if self.val_metrics is not None:
            batch = self.evaluation.apply.batch(batch)
            outputs = self.eval_step(batch)
            for v in self.val_metrics.values():
                preds = getattr(outputs, v["outputs"])
                target = getattr(batch, v["batch"])
                v["metric"](preds, target)

    def validation_epoch_end(self, validation_step_outputs):
        if self.val_metrics is not None:
            for k, v in self.val_metrics.items():
                self.log(f"val/{k}", v["metric"].compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        if self.test_metrics is not None:
            batch = self.evaluation.apply.batch(batch)
            outputs = self.eval_step(batch)
            for v in self.test_metrics.values():
                preds = getattr(outputs, v["outputs"])
                target = getattr(batch, v["batch"])
                v["metric"](preds, target)

    def test_epoch_end(self, validation_step_outputs):
        if self.test_metrics is not None:
            for k, v in self.test_metrics.items():
                self.log(f"test/{k}", v["metric"].compute(), prog_bar=True)
