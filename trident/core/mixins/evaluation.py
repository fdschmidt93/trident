from functools import lru_cache
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence, Union

import hydra
from lightning import LightningModule
from lightning.pytorch.utilities.parsing import AttributeDict
from omegaconf.base import DictKeyType
from omegaconf.dictconfig import DictConfig

from trident.core.datamodule import TridentDataModule
from trident.core.dataset import TridentDataset
from trident.utils import deepgetitem
from trident.utils.enums import Split
from trident.utils.hydra import get_dataset_cfg
from trident.utils.logging import get_logger
from trident.utils.transform import flatten_dict

log = get_logger(__name__)


class EvalMixin(LightningModule):
    hparams: AttributeDict
    log: Callable
    r"""Mixin for base model to define evaluation loop largely via hydra.

    The evaluation mixin enables writing evaluation via yaml files, here is an
    example for sequence classification, borrowed from configs/evaluation/classification.yaml.

    .. code-block:: yaml

        # apply transformation function
        prepare:
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

    def __init__(self) -> None:
        # this line effectively initializes the `LightningModule` of `TridentModule`
        super().__init__()
        # self.hparams is also then accessible in `TridentModule` after super().__init__()
        self.save_hyperparameters()
        evaluation_cfg = self.hparams.get("evaluation")
        assert (
            evaluation_cfg is not None
        ), "Experiment requires appropiate evaluation configuration"
        self.evaluation_cfg = hydra.utils.instantiate(evaluation_cfg)
        # Lightning 2.0 requires manual management of evaluation step outputs
        self._eval_outputs = []

    def log_metric(
        self,
        split: Split,
        metric_key: Union[str, DictKeyType],
        input: Union[int, float, dict],
        log_kwargs: Optional[dict[str, Any]] = None,
        dataset_name: Optional[str] = None,
    ):
        """
        Log a metric for a given split with optional transformation.

        Parameters:
        - split (str): The evaluation split (e.g., "train", "val", "test").
        - metric_key (Union[str, DictKeyType]): Key identifying the metric.
        - input (Union[int, float, dict]): Metric value or dictionary of metric values.
        - log_kwargs (Optional[dict[str, Any]]): Additional keyword arguments for logging.
        - dataset_name (Optional[str]): Name of the dataset if available.

        Notes:
        - This method assumes the existence of `self.evaluation.metrics`.
        - If `input` is a dictionary, each key-value pair is logged separately with the appropriate prefix.
        """
        # Determine if there's a transformation function for the metric
        log_kwargs = log_kwargs or {}
        log_kwargs["prog_bar"] = True

        prefix = dataset_name + "/" + split.value if dataset_name else split.value

        if isinstance(input, dict):
            log_kwargs["dictionary"] = {f"{prefix}/{k}": v for k, v in input.items()}
            self.log_dict(**log_kwargs)
        else:
            log_key = f"{prefix}/{metric_key}"
            self.log(name=log_key, value=input, **log_kwargs)

    @lru_cache
    def _get_configured_function(
        self,
        key: str,
        split: Split,
        dataset_name: Optional[str] = None,
    ) -> Union[None, Callable]:
        """
        Retrieve the configured function from evaluation config based on the provided path and dataset.

        This internal method abstracts away the logic of extracting a function from
        the OmegaConf configuration. It handles both scenarios:
        1. When the function is directly configured.
        2. When the function is configured per dataset.

        Args:
            cfg_path (str):
                The path in the OmegaConf configuration where the function or dataset-specific configurations are stored.
            dataset_name (Optional[str], optional):
                The dataset for which the function needs to be fetched. If not provided, the method assumes
                a direct function configuration. Defaults to None.

        Returns:
            Union[None, Callable]:
                The configured function if resolved; otherwise, None.
        """
        # Select the function or dataset-specific configuration from the OmegaConf configuration using the provided path.
        function_or_config: DictConfig = get_dataset_cfg(
            self.evaluation_cfg.prepare, split=split, dataset_name=dataset_name
        )
        fn: None | Callable = function_or_config.get(key)
        return fn

    def prepare_batch(
        self, split: Split, batch: dict, dataset_name: Optional[str] = None
    ) -> dict:
        """
        Prepares the batch data for a given evaluation split, and, optionally, for a specific dataset.

        Args:
            split (:obj:`str`):
                Evaluation split, such as 'train', 'validation', or 'test'.
            batch (:obj:`dict`):
                Batch data to be prepared.
            dataset_name (:obj:`Optional[str]`, `optional`):
                Name of the dataset, if specified. Default is None.

        Returns:
            :obj:`dict`: Prepared batch data.

        Notes:
            Configuration for this function can be specified in Hydra. If working with a single
            dataset or all datasets have the same configuration, the :obj:`yaml` would look like:

            .. code-block:: yaml

                prepare:
                  batch: null  # takes (module: TridentModule, batch: dict, split: star)
                  outputs:     # takes (module: TridentModule, outputs: dict, batch: dict, split: str)
                    _partial_: true
                    _target_: src.tasks.text_classification.eval.get_preds
                    # takes (module: TridentModule, flattened_step_outputs: dict, split: str)
                    # where list of step_outputs are flattened
                  step_outputs: null

            For many heterogeneous datasets, configuration varies. Each dataset may now require separate configuration!

            .. code-block:: yaml

                prepare:
                  batch: null  # specification not required
                  outputs:
                    _datasets_:
                      validation_ppl:
                        _partial_: true
                        _target_: src.tasks.language_modeling.eval.get_num_tokens
                      validation_nli:
                        _partial_: true
                        _target_: src.tasks.text_classification.eval.get_preds
                      test_nli:
                        _partial_: true
                        _target_: src.tasks.text_classification.eval.get_preds
                  step_outputs: null  # specification not required
        """
        fn = self._get_configured_function(
            key="batch", split=split, dataset_name=dataset_name
        )
        if fn:
            return fn(
                trident_module=self,
                batch=batch,
                split=split,
                dataset_name=dataset_name,
            )
        return batch

    def prepare_outputs(
        self,
        split: Split,
        outputs: dict,
        batch: dict,
        dataset_name: Optional[str] = None,
    ) -> dict:
        """
        Prepares the output data for a given evaluation split, and, optionally, for a specific dataset.

        Args:
            split (:obj:`str`):
                Evaluation split, such as 'train', 'validation', or 'test'.
            outputs (:obj:`dict`):
                Output data to be prepared.
            batch (:obj:`dict`):
                Batch data.
            dataset_name (:obj:`Optional[str]`, `optional`):
                Name of the dataset, if specified. Default is None.

        Returns:
            :obj:`dict`: Prepared outputs data.

        Notes:
            Configuration for this function can be specified in Hydra. Format depends on whether
            working with a single dataset, many homogeneous datasets, or many heterogeneous datasets.
            Refer to the provided examples.
        """
        fn = self._get_configured_function(
            split=split, dataset_name=dataset_name, key="outputs"
        )
        if fn:
            return fn(
                trident_module=self,
                outputs=outputs,
                batch=batch,
                split=split,
                dataset_name=dataset_name,
            )
        return outputs

    def prepare_step_outputs(
        self, split: Split, step_outputs: dict, dataset_name: Optional[str] = None
    ) -> dict:
        """
        Prepares the step outputs for a given evaluation split, and, optionally, for a specific dataset.

        Args:
            split (:obj:`str`):
                Evaluation split, such as 'train', 'validation', or 'test'.
            step_outputs (:obj:`dict`):
                Step outputs data to be prepared.
            dataset_name (:obj:`Optional[str]`, `optional`):
                Name of the dataset, if specified. Default is None.

        Returns:
            :obj:`dict`: Prepared step outputs data.

        Notes:
            Configuration for this function can be specified in Hydra. Format depends on whether
            working with a single dataset, many homogeneous datasets, or many heterogeneous datasets.
            Refer to the provided examples.

            .. seealso:: :py:meth:`trident.core.mixins.evaluation.EvalMixin.prepare_batch`

        """
        fn = self._get_configured_function(
            split=split, dataset_name=dataset_name, key="step_outputs"
        )
        if fn:
            return fn(
                trident_module=self,
                outputs=step_outputs,
                split=split,
                dataset_name=dataset_name,
            )
        return step_outputs

    def _prepare_metric_input(
        self,
        cfg: Union[dict, DictConfig],
        outputs: Union[dict, NamedTuple],
        batch: Optional[Union[dict, NamedTuple]] = None,
    ) -> dict:
        """
        Collects user-defined attributes of outputs & batch to compute a metric.

        The function extracts required variables from `self`, `cfg`, `outputs` and `batch`
        prior to calling the metric.

        .. code-block:: yaml

        # metrics should be copied for each dataset by default unless _datasets_ is specified
        metrics:
          # name of the metric used e.g. for logging
          acc:
            # instructions to instantiate metric, preferably torchmetrics.Metric
            metric:
              _partial_: true
              _target_: torchmetrics.functional.accuracy
              task: "multiclass"
              num_classes: 3
            # either "eval_step" or "epoch_end"
            compute_on: "epoch_end"
            kwargs:
              preds: "outputs:preds"
              target: "outputs:labels"

        Args:
            cfg (Union[dict, DictConfig]): Configuration dictionary for metric computation.
            outputs (Union[dict, NamedTuple]): Outputs data.
            batch (Optional[Union[dict, NamedTuple]]): Batch data.

        Returns:
            dict: Dictionary containing required inputs for metric computation.

        Raises:
            ValueError: If the required key is not found in the provided data.
        """
        ret = {}
        data_sources = {
            "trident_module": self,
            "outputs": outputs,
            "batch": batch,
            "cfg": cfg,
        }
        for k, v in cfg.items():
            source_name, key = v.split(":")
            source_data = data_sources.get(source_name)

            if source_data is None:
                raise ValueError(f"{source_name} not a recognized data source.")

            val = deepgetitem(source_data, key)
            if val is None:
                raise ValueError(f"{key} not found in {source_name} ({source_data}).")

            ret[k] = val

        return ret

    @staticmethod
    def _collect_step_output(
        outputs: dict,
        batch: dict,
        split_dico: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
    ) -> dict:
        """
        Collect user-defined attributes from outputs and batch at the end of `eval_step` into a dictionary.

        This function looks into the provided `split_dico` to determine which attributes from
        `outputs` and `batch` need to be collected. If `split_dico` is not provided, it defaults
        to collecting all attributes.

        .. seealso::
            :py:meth:`trident.core.mixins.evaluation.EvalMixin.eval_step`
            :py:meth:`trident.core.mixins.evaluation.EvalMixin.prepare_step_outputs`

        Args:
            outputs (Dict[str, Any]): Dictionary containing output data.
            batch (Dict[str, Any]): Dictionary containing batch data.
            split_dico (Optional[Union[Dict[str, Union[str, list]], DictConfig]]):
                Dictionary specifying attributes to extract from `outputs` and `batch`.

        Returns:
            Dict[str, Any]: Dictionary of extracted attributes.
        """

        # Handle the absence of split_dico
        if split_dico is None:
            return {"outputs": outputs, "batch": batch}

        ret = {}
        source_mapping = {"outputs": outputs, "batch": batch}
        source_mapping: dict[str, dict[str, Any]] = {"outputs": outputs, "batch": batch}

        for source_key, attribute_keys in split_dico.items():
            # Ensuring type-correctness for the get method
            if not isinstance(source_key, str):
                log.warning(
                    f"Unexpected key type {type(source_key)}. Expected a string."
                )
                continue
            source_data = source_mapping.get(source_key)
            if source_data is None:
                log.warn(
                    f"{source_key} is not supported, can only be one of ['outputs', 'batch']"
                )

            # If attribute_keys is a string, convert to a list for uniform processing
            if isinstance(attribute_keys, str):
                attribute_keys = [attribute_keys]

            # Extract values from the source data
            if source_data is not None:
                for attr_key in attribute_keys:
                    attr_value = source_data.get(attr_key)
                    if attr_value is not None:
                        ret[attr_key] = attr_value
                    else:
                        log.warn(f"{attr_key} not in {source_key}")
        return ret

    @lru_cache
    def _get_dataset_name(
        self, split: Split, dataloader_idx: Optional[int]
    ) -> Optional[str]:
        # Handle multiple datasets.
        dataset_name: None | str = None
        if dataloader_idx is not None:
            datamodule: None | TridentDataModule = getattr(self.trainer, "datamodule")
            assert isinstance(
                datamodule, TridentDataModule
            ), "self.trainer.datamodule not a TridentDataModule! Unsupported operation."
            dataset = datamodule.datasets[split]
            assert (
                dataset is not None
            ), f"dataset for {split} must be incorrectly set up!"
            dataset_name = dataset.key_at_index(dataloader_idx)
        return dataset_name

    def eval_step(
        self, split: Split, batch: dict, dataloader_idx: Optional[int] = None
    ) -> None:
        """Performs model forward & user batch transformation in an eval step.

        This function is called in `validation_step` and `test_step` of the LightningModule.
        """
        dataset_name = self._get_dataset_name(split, dataloader_idx)
        metrics_cfg = get_dataset_cfg(
            self.evaluation_cfg.get("metrics"), split, dataset_name
        )

        # Prepare batch and outputs
        batch = self.prepare_batch(split=split, batch=batch, dataset_name=dataset_name)
        outputs = self.prepare_outputs(
            split, self(batch), batch, dataset_name=dataset_name
        )

        # Compute metrics
        if isinstance(metrics_cfg, Mapping):
            for v in metrics_cfg.values():
                if getattr(v, "compute_on", False) == "eval_step":
                    kwargs = self._prepare_metric_input(v.kwargs, outputs, batch)
                    v["metric"](**kwargs)

        # Handle step outputs collection
        dataloader_idx = dataloader_idx or 0  # Default to 0 if None
        try:
            eval_outputs = self._eval_outputs[dataloader_idx]
        except IndexError:
            if len(self._eval_outputs) == dataloader_idx:
                eval_outputs = []
                self._eval_outputs.append(eval_outputs)
            else:
                raise IndexError(
                    f"dataloader_idx {dataloader_idx} is not subsequent index in evaluation, check warranted!"
                )

        step_collection_cfg = get_dataset_cfg(
            self.evaluation_cfg.get("step_outputs"), split, dataset_name
        )
        if step_collection_cfg is None or isinstance(step_collection_cfg, Mapping):
            eval_outputs.append(
                self._collect_step_output(outputs, batch, step_collection_cfg)
            )

    def _evaluate_metrics_for_dataset(
        self,
        split: Split,
        step_outputs: list[dict],
        metrics_cfg: DictConfig,
        dataset_name: Optional[str] = None,
    ) -> None:
        """Compute and log metrics at the epoch's end for a specified dataset.

        Args:
            split (str): Evaluation split, e.g., "val" or "test".
            step_outputs (List[Dict]): Aggregated step outputs for the dataset.
            metrics_cfg (DictConfig): Metric configurations for the dataset.
            dataset_name (Optional[str]): Name of the dataset.
        """
        flattened_step_outputs = flatten_dict(step_outputs)
        prepared_outputs = self.prepare_step_outputs(
            split, flattened_step_outputs, dataset_name
        )
        for metric, metric_cfg in metrics_cfg.items():
            if getattr(metric_cfg, "compute_on", False) == "eval_step":
                input_ = metric_cfg["metric"]
            elif getattr(metric_cfg, "compute_on", "epoch_end") == "epoch_end":
                kwargs = self._prepare_metric_input(metric_cfg.kwargs, prepared_outputs)
                input_ = metric_cfg["metric"](**kwargs)
            else:
                raise ValueError(
                    f"`compute_on` of metric_cfg ({getattr(metric_cfg, 'compute_on')}) is not one of ['eval_step', 'epoch_end']"
                )
            self.log_metric(
                split=split,
                metric_key=metric,
                input=input_,
                dataset_name=dataset_name,
            )

    def on_eval_epoch_end(self, split: Split) -> None:
        """Compute and log metrics for all datasets at the epoch's end.

        Note: the epoch only ends when all datasets are processed.

        This method determines if multiple datasets exist for the evaluation split and
        appropriately logs the metrics for each.

        Args:
            split (str): Evaluation split, e.g., "val" or "test".
        """
        # `metrics_cfg` is always the fallback, means lower level of hydra config does not exist
        step_outputs: list[list[dict]] = self._eval_outputs
        datamodule = getattr(self.trainer, "datamodule")
        assert isinstance(
            datamodule, TridentDataModule
        ), "datamodule must be `TridentDataModule`!"
        datasets: None | TridentDataset = datamodule.datasets[split]
        if datasets is not None:
            # idx aligns with dataloader_idx (i.e., sequential order of eval datasets) of lightning
            for idx, dataset_name in enumerate(datasets.keys()):
                if dataset_metrics := get_dataset_cfg(
                    self.evaluation_cfg.get("metrics"), split, dataset_name
                ):
                    self._evaluate_metrics_for_dataset(
                        split=split,
                        step_outputs=step_outputs[idx],
                        metrics_cfg=dataset_metrics,
                        dataset_name=dataset_name,
                    )
        self._eval_outputs.clear()

    def validation_step(
        self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> None:
        return self.eval_step(Split.VAL, batch, dataloader_idx)

    def on_validation_epoch_end(self):
        return self.on_eval_epoch_end(Split.VAL)

    def test_step(
        self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> None:
        return self.eval_step(Split.TEST, batch, dataloader_idx)

    def on_test_epoch_end(self):
        return self.on_eval_epoch_end(Split.TEST)
