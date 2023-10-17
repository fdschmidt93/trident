from typing import Any, Callable, NamedTuple, Optional, Union

import hydra
from lightning import LightningModule
from lightning.pytorch.utilities.parsing import AttributeDict
from omegaconf import OmegaConf
from omegaconf.base import DictKeyType
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset

from trident.core.datamodule import TridentDataModule
from trident.utils import deepgetitem
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

    def __init__(self) -> None:
        # hparams used to fast-forward required attributes
        evaluation_cfg = self.hparams.get("evaluation")
        assert (
            evaluation_cfg is not None
        ), "Experiment requires appropiate evaluation configuration"
        self.evaluation = hydra.utils.instantiate(evaluation_cfg)
        # Lightning 2.0 requires manual management of evaluation step outputs
        self._eval_outputs = []

    def on_eval_start(self, stage: str) -> None:
        """
        Prepare the metrics configurations and move metrics to the appropriate device at the beginning of evaluation.

        This method handles metrics configuration at the start of evaluation for a given stage.
        It ensures that all metrics are moved to the appropriate device. Additionally, for datasets
        that have specific configurations, this method deepcopies the original metrics configuration
        for each dataset, ensuring each dataset has its own configuration.

        Parameters:
        - stage (str): The evaluation stage, e.g., "train", "val", "test".

        Returns:
        - None

        Notes:
        - This method assumes the existence of `self.evaluation.metrics_cfg` and `self.trainer.datamodule`.
        - Metrics if PyTorch objects like `torchmetrics` OOP API are moved between devices.
        - The method is private and should not be called outside of this class.
        """
        metrics_cfg = self.evaluation.metrics_cfg
        if metrics_cfg is None:
            return

        stage_metrics_cfg: Union[None, dict[str, Any]] = metrics_cfg.get(stage, None)
        dataset: Union[Dataset, dict[str, Dataset]] = getattr(self.trainer.datamodule, f"dataset_{stage}")  # type: ignore

        if stage_metrics_cfg is not None:
            configs = stage_metrics_cfg.get("_datasets_", {"val": stage_metrics_cfg})

            for cfg in configs.values():
                self._move_metrics_to_device(cfg)

            # Deepcopy original metrics cfg for each dataset
            # TODO(fdschmidt93): Should be part of config_callbacks but torchmetrics OOP API requires later merging
            if isinstance(dataset, dict) and "_datasets_" not in stage_metrics_cfg:
                self._deepcopy_metrics_for_each_dataset(
                    dataset, stage, stage_metrics_cfg
                )

    def _move_metrics_to_device(self, cfg: dict[str, Any]) -> None:
        """Move metrics to the current device for torchmetrics OOP API."""
        for metric_cfg in cfg.values():
            metric = metric_cfg["metric"]
            # torchmetrics OOP API
            if hasattr(metric, "to") and hasattr(metric, "device"):
                metric.to(self.device)

    def _deepcopy_metrics_for_each_dataset(
        self, dataset: dict[str, Dataset], stage: str, stage_metrics_cfg: dict[str, Any]
    ) -> None:
        """Deepcopy metrics config for each dataset."""
        self.evaluation.metrics_cfg[stage] = {}
        self.evaluation.metrics_cfg[stage]["_datasets_"] = {}

        for name in dataset:
            if self.hparams.evaluation is not None and hasattr(
                self.hparams.evaluation, "metrics_cfg"
            ):
                self.evaluation.metrics_cfg[stage]["_datasets_"][
                    name
                ] = hydra.utils.instantiate(
                    self.hparams.evaluation.metrics_cfg.get(stage)
                )
                metrics: dict[str, Any] = self.evaluation.metrics_cfg[stage][
                    "_datasets_"
                ][name]
                self._move_metrics_to_device(metrics)

    def on_validation_start(self) -> None:
        self.on_eval_start(stage="val")

    def on_test_start(self) -> None:
        self.on_eval_start(stage="test")

    def log_metric(
        self,
        stage: str,
        metric_key: Union[str, DictKeyType],
        input: Union[int, float, dict],
        log_kwargs: Optional[dict[str, Any]] = None,
        dataset_name: Optional[str] = None,
    ):
        """
        Log a metric for a given stage with optional transformation.

        Parameters:
        - stage (str): The evaluation stage (e.g., "train", "val", "test").
        - metric_key (Union[str, DictKeyType]): Key identifying the metric.
        - input (Union[int, float, dict]): Metric value or dictionary of metric values.
        - log_kwargs (Optional[dict[str, Any]]): Additional keyword arguments for logging.
        - dataset_name (Optional[str]): Name of the dataset if available.

        Notes:
        - This method assumes the existence of `self.evaluation.metrics_cfg`.
        - If `input` is a dictionary, each key-value pair is logged separately with the appropriate prefix.
        """
        # Determine if there's a transformation function for the metric
        log_kwargs = log_kwargs or {}
        log_kwargs["prog_bar"] = True

        prefix = dataset_name + "/" + stage if dataset_name else stage

        if isinstance(input, dict):
            log_kwargs["dictionary"] = {f"{prefix}/{k}": v for k, v in input.items()}
            self.log_dict(**log_kwargs)
        else:
            log_key = f"{prefix}/{metric_key}"
            self.log(name=log_key, value=input, **log_kwargs)

    def _get_configured_function(
        self, cfg_path: str, dataset: Optional[str] = None
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
            dataset (Optional[str], optional):
                The dataset for which the function needs to be fetched. If not provided, the method assumes
                a direct function configuration. Defaults to None.

        Returns:
            Union[None, Callable]:
                The configured function if resolved; otherwise, None.
        """
        # Select the function or dataset-specific configuration from the OmegaConf configuration using the provided path.
        function_or_config: Union[None, Callable, DictConfig] = OmegaConf.select(
            self.evaluation, cfg_path
        )

        # If dataset is provided and the selected configuration is a DictConfig,
        # retrieve the function specific to that dataset.
        if (
            function_or_config
            and isinstance(function_or_config, DictConfig)
            and dataset is not None
        ):
            function_or_config = function_or_config._datasets_.get(dataset)

        # If the final selected configuration is callable (i.e., a function), return it.
        if isinstance(function_or_config, Callable):
            return function_or_config

        # If no function is found, return None.
        return None

    # TODO(fdschmidt93): can we reduce overhead?
    def prepare_batch(
        self, stage: str, batch: dict, dataset: Optional[str] = None
    ) -> dict:
        """
        Prepares the batch data for a given evaluation stage, and, optionally, for a specific dataset.

        Args:
            stage (:obj:`str`):
                Evaluation stage, such as 'train', 'validation', or 'test'.
            batch (:obj:`dict`):
                Batch data to be prepared.
            dataset (:obj:`Optional[str]`, `optional`):
                Name of the dataset, if specified. Default is None.

        Returns:
            :obj:`dict`: Prepared batch data.

        Notes:
            Configuration for this function can be specified in Hydra. If working with a single
            dataset or all datasets have the same configuration, the :obj:`yaml` would look like:

            .. code-block:: yaml

                prepare_cfg:
                  batch: null  # takes (module: TridentModule, batch: dict, stage: star)
                  outputs:     # takes (module: TridentModule, outputs: dict, batch: dict, stage: str)
                    _partial_: true
                    _target_: src.tasks.text_classification.eval.get_preds
                    # takes (module: TridentModule, flattened_step_outputs: dict, stage: str)
                    # where list of step_outputs are flattened
                  step_outputs: null

            For many heterogeneous datasets, configuration varies. Each dataset may now require separate configuration!

            .. code-block:: yaml

                prepare_cfg:
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
        fn = self._get_configured_function(f"prepare_cfg.{stage}.batch", dataset)
        if fn:
            return fn(self, batch=batch, stage=stage)
        return batch

    def prepare_outputs(
        self, stage: str, outputs: dict, batch: dict, dataset: Optional[str] = None
    ) -> dict:
        """
        Prepares the output data for a given evaluation stage, and, optionally, for a specific dataset.

        Args:
            stage (:obj:`str`):
                Evaluation stage, such as 'train', 'validation', or 'test'.
            outputs (:obj:`dict`):
                Output data to be prepared.
            batch (:obj:`dict`):
                Batch data.
            dataset (:obj:`Optional[str]`, `optional`):
                Name of the dataset, if specified. Default is None.

        Returns:
            :obj:`dict`: Prepared outputs data.

        Notes:
            Configuration for this function can be specified in Hydra. Format depends on whether
            working with a single dataset, many homogeneous datasets, or many heterogeneous datasets.
            Refer to the provided examples.
        """
        fn = self._get_configured_function(f"prepare_cfg.{stage}.outputs", dataset)
        if fn:
            return fn(self, outputs=outputs, batch=batch, stage=stage)
        return outputs

    def prepare_step_outputs(
        self, stage: str, step_outputs: dict, dataset: Optional[str] = None
    ) -> dict:
        """
        Prepares the step outputs for a given evaluation stage, and, optionally, for a specific dataset.

        Args:
            stage (:obj:`str`):
                Evaluation stage, such as 'train', 'validation', or 'test'.
            step_outputs (:obj:`dict`):
                Step outputs data to be prepared.
            dataset (:obj:`Optional[str]`, `optional`):
                Name of the dataset, if specified. Default is None.

        Returns:
            :obj:`dict`: Prepared step outputs data.

        Notes:
            Configuration for this function can be specified in Hydra. Format depends on whether
            working with a single dataset, many homogeneous datasets, or many heterogeneous datasets.
            Refer to the provided examples.

            .. seealso:: :py:meth:`trident.core.mixins.evaluation.EvalMixin.prepare_batch`

        """
        fn = self._get_configured_function(f"prepare_cfg.{stage}.step_outputs", dataset)
        if fn:
            return fn(self, outputs=step_outputs, stage=stage, dataset=dataset)
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

        # metrics_cfg should be copied for each dataset by default unless _datasets_ is specified
        metrics_cfg:
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
        data_sources = {"self": self, "outputs": outputs, "batch": batch, "cfg": cfg}

        for k, v in cfg.items():
            source_name, key = v.split(":")
            source_data = data_sources.get(source_name)

            if source_data is None:
                raise ValueError(f"{source_name} not a recognized data source.")

            val = deepgetitem(source_data, key)
            if val is None:
                raise ValueError(f"{k} not found in {source_name}.")

            ret[k] = val

        return ret

    @staticmethod
    def _collect_step_output(
        outputs: dict,
        batch: dict,
        stage_dico: Optional[
            Union[dict[str, Union[str, list[str]]], DictConfig]
        ] = None,
    ) -> dict:
        """
        Collect user-defined attributes from outputs and batch at the end of `eval_step` into a dictionary.

        This function looks into the provided `stage_dico` to determine which attributes from
        `outputs` and `batch` need to be collected. If `stage_dico` is not provided, it defaults
        to collecting all attributes.

        .. seealso::
            :py:meth:`trident.core.mixins.evaluation.EvalMixin.eval_step`
            :py:meth:`trident.core.mixins.evaluation.EvalMixin.prepare_step_outputs`

        Args:
            outputs (Dict[str, Any]): Dictionary containing output data.
            batch (Dict[str, Any]): Dictionary containing batch data.
            stage_dico (Optional[Union[Dict[str, Union[str, list]], DictConfig]]):
                Dictionary specifying attributes to extract from `outputs` and `batch`.

        Returns:
            Dict[str, Any]: Dictionary of extracted attributes.
        """

        # Handle the absence of stage_dico
        if stage_dico is None:
            return {"outputs": outputs, "batch": batch}

        ret = {}
        source_mapping = {"outputs": outputs, "batch": batch}
        source_mapping: dict[str, dict[str, Any]] = {"outputs": outputs, "batch": batch}

        for source_key, attribute_keys in stage_dico.items():
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

    def eval_step(
        self, stage: str, batch: dict, dataloader_idx: Optional[int] = None
    ) -> None:
        """Performs model forward & user batch transformation in an eval step.

        This function is called in `validation_step` and `test_step` of the LightningModule.
        """

        metrics_cfg: None | DictConfig = self.evaluation.metrics_cfg
        if not metrics_cfg:
            return

        # Retrieve stage-specific configuration or default to base config.
        metrics_cfg = metrics_cfg.get(stage, metrics_cfg)

        # Handle multiple datasets.
        dataset: None | str = None
        if dataloader_idx is not None:
            datamodule: None | TridentDataModule = getattr(self.trainer, "datamodule")
            assert isinstance(
                datamodule, TridentDataModule
            ), f"self.trainer.datamodule not a TridentDataModule! Unsupported operation."
            idx2dataset: dict[int, str] = getattr(datamodule, f"idx2dataset_{stage}")
            dataset = idx2dataset[dataloader_idx]

        if metrics_cfg and "_datasets_" in metrics_cfg:
            assert dataset is not None  # Ensure dataset was successfully fetched
            metrics_cfg = metrics_cfg["_datasets_"].get(dataset)

        # Prepare batch and outputs
        batch = self.prepare_batch(stage=stage, batch=batch, dataset=dataset)
        outputs = self.prepare_outputs(stage, self(batch), batch, dataset=dataset)

        # Compute metrics
        if metrics_cfg:
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

        step_collection_dico = OmegaConf.select(
            self.evaluation, f"step_outputs.{stage}"
        )
        if dataset and step_collection_dico and "_datasets_" in step_collection_dico:
            step_collection_dico = step_collection_dico["_datasets_"].get(dataset)

        eval_outputs.append(
            self._collect_step_output(outputs, batch, step_collection_dico)
        )

    def _evaluate_metrics_for_dataset(
        self,
        stage: str,
        step_outputs: list[dict],
        metrics_cfg: DictConfig,
        dataset_name: Optional[str] = None,
    ) -> None:
        """Compute and log metrics at the epoch's end for a specified dataset.

        Args:
            stage (str): Evaluation stage, e.g., "val" or "test".
            step_outputs (List[Dict]): Aggregated step outputs for the dataset.
            metrics_cfg (DictConfig): Metric configurations for the dataset.
            dataset_name (Optional[str]): Name of the dataset.
        """
        flattened_step_outputs = flatten_dict(step_outputs)
        prepared_outputs = self.prepare_step_outputs(
            stage, flattened_step_outputs, dataset_name
        )

        for metric, metric_cfg in metrics_cfg.items():
            if getattr(metric_cfg, "compute_on", False) == "eval_step":
                self.log_metric(
                    stage=stage, metric_key=metric, input=metric_cfg["metric"]
                )
            elif getattr(metric_cfg, "compute_on", False) == "epoch_end":
                kwargs = self._prepare_metric_input(metric_cfg.kwargs, prepared_outputs)
                self.log_metric(
                    stage=stage,
                    metric_key=metric,
                    input=metric_cfg["metric"](**kwargs),
                    dataset_name=dataset_name,
                )

    def on_eval_epoch_end(self, stage: str) -> None:
        """Compute and log metrics for all datasets at the epoch's end.

        This method determines if multiple datasets exist for the evaluation stage and
        appropriately logs the metrics for each.

        Args:
            stage (str): Evaluation stage, e.g., "val" or "test".
        """
        metrics_cfg = self.evaluation.metrics_cfg.get(
            stage, self.evaluation.metrics_cfg
        )
        if not metrics_cfg:
            return

        step_outputs: list[list[dict]] = self._eval_outputs
        datamodule = getattr(self.trainer, "datamodule")
        assert isinstance(
            datamodule, TridentDataModule
        ), "datamodule must be `TridentDataModule`!"
        idx2dataset = getattr(datamodule, f"idx2dataset_{stage}", None)

        if idx2dataset:
            for idx, dataset_name in idx2dataset.items():
                dataset_metrics = metrics_cfg["_datasets_"].get(dataset_name)
                if dataset_metrics:
                    self._evaluate_metrics_for_dataset(
                        stage=stage,
                        step_outputs=step_outputs[idx],
                        metrics_cfg=dataset_metrics,
                        dataset_name=dataset_name,
                    )
        else:
            assert len(step_outputs) == 1, "Expected step outputs for only one dataset."
            self._evaluate_metrics_for_dataset(
                stage=stage,
                step_outputs=step_outputs[0],
                metrics_cfg=metrics_cfg,
            )

        self._eval_outputs.clear()

    def validation_step(
        self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> None:
        return self.eval_step("val", batch, dataloader_idx)

    def on_validation_epoch_end(self):
        return self.on_eval_epoch_end("val")

    def test_step(
        self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> None:
        return self.eval_step("test", batch, dataloader_idx)

    def on_test_epoch_end(self):
        return self.on_eval_epoch_end("test")
