from typing import Callable, NamedTuple, Optional, Union

import hydra
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.parsing import AttributeDict

from trident.utils.transform import flatten_dict

# TODO(fdschmidt93): update docs
# TODO(fdschmidt93): validation
# TODO(fdschmidt93): what if no stage?


class EvalMixin(LightningModule):
    r"""Mixin for base model to define evaluation loop largely via hydra.

    The evaluation mixin enables writing evaluation via yaml files, here is an
    example for sequence classification, borrowed from configs/evaluation/classification.yaml.

    .. code-block:: yaml

        # apply transformation function
        prepare_cfg:
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
        metrics_cfg:
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

    def logging(
        self,
        stage: str,
        metric_key: str,
        input: Union[int, float, dict],
        log_kwargs: Optional[dict] = None,
    ):
        fn: Union[None, Callable] = OmegaConf.select(
            self.evaluation, f"metrics_cfg.{stage}.{metric_key}.logging"
        )
        if fn is not None:
            input = fn(input)
        log_kwargs = log_kwargs if log_kwargs is not None else {}

        if isinstance(input, dict):
            # MAYBE(fdschmidt93): better default formatting? might be taken care by pytorch_lightning itself
            log_kwargs["dictionary"] = input
            self.log_dict(**log_kwargs)
        else:
            log_kwargs["name"] = f"{stage}/{metric_key}"
            log_kwargs["value"] = input
            self.log(**log_kwargs)

    # TODO(fdschmidt93): can we reduce overhead?
    def prepare_batch(self, stage: str, batch: dict) -> dict:
        fn: Union[None, Callable] = OmegaConf.select(
            self.evaluation, f"prepare_cfg.{stage}.batch"
        )
        if fn is not None:
            return fn(self, batch, stage)
        return batch

    def prepare_outputs(self, stage: str, outputs: dict, batch: dict) -> dict:
        fn: Union[None, Callable] = OmegaConf.select(
            self.evaluation, f"prepare_cfg.{stage}.outputs"
        )
        if fn is not None:
            return fn(self, outputs, batch, stage)
        return outputs

    def prepare_step_outputs(self, stage: str, step_outputs: dict) -> dict:
        fn: Union[None, Callable] = OmegaConf.select(
            self.evaluation, f"prepare_cfg.{stage}.step_outputs"
        )
        if fn is not None:
            return fn(self, step_outputs, stage)
        return step_outputs

    # TODO(fdschmidt93): switch from `locals` to kwargs?
    def _prepare_metric_input(
        self,
        cfg: DictConfig,
        outputs: Union[dict, NamedTuple],
        batch: Optional[Union[dict, NamedTuple]] = None,
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
        # TODO(fdschmidt93): allow self
        ret = {}
        local_vars = locals()
        for k, v in cfg.items():
            var, key = v.split(":")
            # TODO(fdschmidt93): rfc
            input_: dict = local_vars.get(var, {})
            val = (
                input_.get(key, None)
                if isinstance(input_, dict)
                else getattr(input_, key, None)
            )
            if val is not None:
                ret[k] = val
            else:
                raise AssertionError(f"{k} not found in {var}")
        return ret

    def _collect_step_output(
        self,
        stage: str,
        outputs: dict,
        batch: dict,
    ) -> dict:
        """Collects user-defined attributes of outputs & batch at end of eval_step in dict."""
        # TODO(fdschmidt93): validate uniqueness
        # TODO(fdschmidt93): enable putting to other device
        # TODO(fdschmidt93): define clear behaviour if no step_outputs is defined
        # TODO(fdschmidt93): restricting step_output arguments to function arguments via inspect library
        stage_dico: Union[None, DictConfig] = OmegaConf.select(
            self.evaluation, f"step_outputs.{stage}"
        )
        if stage_dico is not None:
            ret = {}
            local_vars = locals()

            def set_val(dico, key, val):
                ret_val = local_vars.get(key, {}).get(val, None)
                if ret_val is None:
                    raise AttributeError(f"{val} not in {key}")
                dico[val] = ret_val

            for key, vals in stage_dico.items():
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

    def eval_step(self, stage: str, batch: dict) -> dict:
        """Performs model forward & user batch transformation in an eval step."""

        batch = self.prepare_batch(stage, batch)
        outputs = self.prepare_outputs(stage, self(batch), batch)

        metrics_cfg = OmegaConf.select(self.evaluation.metrics_cfg, stage)
        if metrics_cfg is not None:
            for v in metrics_cfg.values():
                if getattr(v, "compute_on", False) == "eval_step":
                    kwargs = self._prepare_metric_input(v.kwargs, outputs, batch)
                    v["metric"](**kwargs)
        return self._collect_step_output(stage, outputs, batch)

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
        flattened_step_outputs = flatten_dict(step_outputs)
        flattened_step_outputs = self.prepare_step_outputs(
            stage, flattened_step_outputs
        )
        metrics_cfg = OmegaConf.select(self.evaluation.metrics_cfg, stage)
        if metrics_cfg is not None:
            for k, v in metrics_cfg.items():
                if getattr(v, "compute_on", False) == "eval_step":
                    # TODO(fdschmidt93): do not rely on having to call `compute` here
                    self.logging(stage=stage, metric_key=k, input=v["metric"].compute())
                if getattr(v, "compute_on", False) == "epoch_end":
                    kwargs: dict = self._prepare_metric_input(
                        v.kwargs, flattened_step_outputs, None
                    )
                    self.logging(stage=stage, metric_key=k, input=v["metric"](**kwargs))
        return flattened_step_outputs

    def validation_step(self, batch: dict, batch_idx: int) -> Union[None, dict]:
        return self.eval_step("val", batch)

    def validation_epoch_end(self, validation_step_outputs: list[dict]):
        return self.eval_epoch_end("val", validation_step_outputs)

    def test_step(self, batch: dict, batch_idx: int) -> Union[None, dict]:
        return self.eval_step("test", batch)

    def test_epoch_end(self, test_step_outputs: list[dict]):
        return self.eval_epoch_end("test", test_step_outputs)
