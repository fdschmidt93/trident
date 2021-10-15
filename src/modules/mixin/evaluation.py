from typing import Callable, Union

import hydra
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from pytorch_lightning.utilities.parsing import AttributeDict
from transformers.file_utils import ModelOutput
from transformers.tokenization_utils_base import BatchEncoding

from src.utils import flatten_dict


class EvalMixin:
    r"""Mixin for base model to define evaluation loop largely via hydra.

    The evaluation mixin enables writing evaluation via yaml files, here is an
    example for sequence classification, borrowed from configs/evaluation/classification.yaml.

    .. code-block:: yaml

        # apply transformation function 
        apply:
          batch: null # on each step
          outputs:    # on each step
            _target_: src.utils.hydra.partial
            _partial_: src.evaluation.classification.get_preds
            .. code-block: python
                
                # we link evaluation.apply.outputs against get_preds
                def get_preds(outputs):
                    outputs.preds = outputs.logits.argmax(dim=-1)
                    return outputs

          step_outputs: null  # on flattened outputs of what's collected from steps

        # Which keys/attributes are supposed to be collected from `outputs` and `batch`
        step_outputs:
          outputs: "preds" # can be a str
          batch: # or a list[str]
            - labels
        # either metrics or val_metrics and test_metrics
        # where the latter
        metrics:
          # name of the metric used eg for logging
          accuracy:
            # instructions to instantiate metric, preferrably torchmetrics.Metric
            metric:
              _target_: torchmetrics.Accuracy
            # either on_step: true or on_epoch: true
            on_step: true
            compute: 
              preds: "outputs:preds"
              target: "batch:labels"
          f1:
            metric:
              _target_: torchmetrics.F1
            on_step: true
            compute:
              preds: "outputs:preds"
              target: "batch:labels"

    """

    hparams: AttributeDict
    log: Callable

    def __init__(self) -> None:

        # hparams used to fast-forward required attributes
        self.evaluation = hydra.utils.instantiate(self.hparams.evaluation)

        for attr in ["batch", "outputs", "step_outputs"]:
            if not callable(getattr(self.evaluation.apply, attr, None)):
                setattr(self.evaluation.apply, attr, None)

        self.metrics: DictConfig = getattr(self.evaluation, "metrics")

    # TODO(fdschmidt93): switch from `locals` to kwargs?
    def prepare_metric_input(
        self,
        outputs: ModelOutput,
        batch: Union[None, dict, BatchEncoding],
        cfg: DictConfig,
    ) -> dict:
        """Collects user-defined attributes of outputs & batch to compute metric.


        Args:
            outputs: 
            batch: [TODO:description]
            cfg: [TODO:description]

        Returns:
            dict: [TODO:description]

        Raises:
            AssertionError: [TODO:description]
        """
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
        # TODO(fdschmidt93): restricting step_output arguments to function arguments via inspect library
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

    def eval_step(self, batch: Union[dict, BatchEncoding]) -> dict:
        """Performs model forward & user batch transformation in an eval step."""
        if self.evaluation.apply.batch is not None:
            batch = self.evaluation.apply.batch(batch)
        outputs = self(batch)
        if self.evaluation.apply.outputs is not None:
            outputs = self.evaluation.apply.outputs(outputs, batch)
        for v in self.metrics.values():
            if getattr(v, "on_step", False):
                kwargs = self.prepare_metric_input(outputs, batch, v.compute)
                v["metric"](**kwargs)
        return self.collect_step_output(outputs, batch)

    def eval_epoch_end(self, stage: str, step_outputs: list[dict]) -> dict:
        """Computes evaluation metric at epoch end for respective `stage`.
        
        Flattening step outputs attempts to stack numpy arrays and tensors along 0 axis.

        Args:
            stage: typically either 'val' or 'test', affects logging
            step_outputs: outputs of eval steps & flattened at start of `eval_epoch_end`

        Returns:
            dict: flattened outputs from evaluation steps
        """
        # if self.metrics is not None:
        outputs = flatten_dict(step_outputs)
        if self.evaluation.apply.step_outputs is not None:
            outputs = self.evaluation.apply.step_outputs(outputs)
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
