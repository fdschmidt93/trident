from pytorch_lightning import LightningModule
from transformers.modeling_outputs import SequenceClassifierOutput


class EvalMixin(LightningModule):
    def validation_step(self, batch, batch_idx):
        if self.val_metrics is not None:
            outputs = self(batch)
            if hasattr(outputs, "logits"):
                outputs.preds = outputs.logits.argmax(-1)
            for attr, metric in self.val_metrics.values():
                metric(getattr(outputs, attr), batch["labels"])

    def validation_epoch_end(self, validation_step_outputs):
        if self.val_metrics is not None:
            for metric, (_, compute_metric) in self.val_metrics.items():
                self.log(f"val/{metric}", compute_metric.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        if self.test_metrics is not None:
            outputs = self(batch)
            if hasattr(outputs, "logits"):
                outputs.preds = outputs.logits.argmax(-1)
            for attr, metric in self.test_metrics.values():
                metric(getattr(outputs, attr), batch["labels"])

    def test_epoch_end(self, test_step_outputs):
        if self.test_metrics is not None:
            for metric, (_, compute_metric) in self.test_metrics.items():
                self.log(f"test/{metric}", compute_metric.compute(), prog_bar=True)
