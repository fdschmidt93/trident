from typing import Callable, Union

import hydra
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.parsing import AttributeDict
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutput
from transformers.tokenization_utils_base import BatchEncoding

from src.utils import flatten_dict


class EvalMixin(LightningModule):
    hparams: AttributeDict
    log: Callable

    """
    Definitions:
        - outputs, ModelOutput: from eg Huggingface transformer model
        - batch, Union[dict, BatchEncoding]: output from HF tokenizer

    evaluation, DictConfig:
        - apply, DictConfig: pointer to function to preprocess ...
            - batch, eval_step: any transformation needed in eval_step
            - outputs, eval_step: any transformation needed in eval_step, e.g.
                                  outputs.preds = outputs.logits.argmax(dim=-1)
            - step_outputs, eval_step: any transformation needed in on_epoch_end
                step_outputs are flattened collected attributes of step_outputs

        - step_outputs, DictConfig: returns a dict of collected attributes
            - batch, DictConfig[Union[str, list[str]]: 
                str or list[str] of which attributes to collect from batch
            - outputs, DictConfig[Union[str, list[str]]: 
                str or list[str] of which attributes to collect from outputs
        - metrics, DictConfig: DictConfigs of metrics to be computed
            - `key`: defines name of metric to be logged
                metric: defines instructions for hydra.utils.instantiate
                on_step|on_epoch, bool: one of on_step or on_epoch needs to be true
                compute, DictConfig: define kwargs for metric to be computed
                    eg for torchmetric.Accuracy
                      preds: "outputs:preds"
                      target: "batch:labels"
                    where syntax "variable:key_of_variable"
    """

    def __init__(self) -> None:
        super().__init__()

        self.evaluation = hydra.utils.instantiate(self.hparams.evaluation)

        # pass identity if transform is not set
        for attr in ["batch", "outputs", "epoch"]:
            if not callable(getattr(self.evaluation.apply, attr, None)):
                setattr(self.evaluation.apply, attr, lambda x: x)

        self.metrics: DictConfig = getattr(self.evaluation, "metrics")

    def prepare_metric_input(
        self,
        outputs: ModelOutput,
        batch: Union[None, dict, BatchEncoding],
        cfg: DictConfig,
    ) -> dict:
        """Collects user-defined attributes of outputs & batch to compute metric."""
        ret = {}
        local_vars = locals()
        for k, v in cfg.items():
            var, key = v.split(":")
            input_ = local_vars.get(var)
            val = None
            if input_ is not None:
                try:
                    val = getattr(input_, key)
                except:
                    val = input_.get(key)
            if val is not None:
                ret[k] = val
            else:
                raise AssertionError(f"{k} not found in {var}")
        return ret

    def collect_step_output(
        self, outputs: ModelOutput, batch: Union[dict, BatchEncoding]
    ) -> dict:
        """Collects user-defined attributes of outputs & batch at end of eval_step in dict."""
        # TODO(fdschmidt93): validate uniqueness
        # TODO(fdschmidt93): enable putting to other device
        # TODO(fdschmidt93): define clear behaviour if no step_outputs is defined
        if self.evaluation.step_outputs is not None:
            ret = {}
            local_vars = locals()

            def set_val(dico, key, val):
                ret_val = getattr(local_vars.get(key), val, None)
                if ret_val is None:
                    raise AttributeError(f"{val} not in {key}")
                dico[val] = ret_val

            for key, vals in self.evaluation.step_outputs.items():
                if isinstance(vals, (ListConfig, list)):
                    for val in vals:
                        set_val(ret, key, val)
                elif isinstance(vals, str):
                    set_val(ret, key, vals)
                else:
                    raise TypeError(
                        f"Should be either str or list[str], not {type(vals)}"
                    )
            return ret
        return {"outputs": outputs, "batch": batch}

        - logits
        - loss
        - last_hidden_state
    def eval_step(self, batch: Union[dict, BatchEncoding]) -> dict:
        """Performs model forward & user batch transformation in an eval step."""
        # return self.evaluation.apply.outputs(self(batch), batch)
        batch = self.evaluation.apply.batch(batch)
        outputs = self.evaluation.apply.outputs(self(batch), batch)
        for v in self.metrics.values():
            if getattr(v, "on_step", False):
                kwargs = self.prepare_metric_input(outputs, batch, v.compute)
                v["metric"](**kwargs)
        return self.collect_step_output(outputs, batch)

    def eval_epoch_end(self, stage: str, step_outputs: list[dict]) -> dict:
        # if self.metrics is not None:
        flattened_outputs = flatten_dict(step_outputs)
        outputs = self.evaluation.apply.step_outputs(flattened_outputs)
        for k, v in self.metrics.items():
            if getattr(v, "on_step", False):
                self.log(f"{stage}/{k}", v["metric"].compute(), prog_bar=True)
            if getattr(v, "on_epoch", False):
                kwargs: dict = self.prepare_metric_input(outputs, None, v.compute)
                self.log(f"{stage}/{k}", v["metric"](**kwargs), prog_bar=True)
        return outputs

    def validation_step(self, batch, batch_idx) -> Union[None, dict]:
        return self.eval_step(batch)

    def validation_epoch_end(self, validation_step_outputs: list[dict]):
        return self.eval_epoch_end("val", validation_step_outputs)

    def test_step(self, batch, batch_idx) -> Union[None, dict]:
        return self.eval_step(batch)

    def test_epoch_end(self, test_step_outputs: list[dict]):
        return self.eval_epoch_end("test", test_step_outputs)


# import torchmetrics
# import torch
# x = torch.randn(1000, 300)
# y = torch.randn(1000, 300)
# p = x @ y.T
# label = torch.zeros(1000, 1000).long()
# idx = torch.arange(1000).repeat(1000, 1)
# idx.fill_diagonal_(1)
# mrr = torchmetrics.RetrievalMRR()
# mrr(p, label, idx) # weird, easier with functional interface
