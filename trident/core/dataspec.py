from copy import deepcopy
from typing import Optional, Any, Sequence
import hydra
from omegaconf.dictconfig import DictConfig
from torch.utils.data.dataloader import DataLoader

from trident.utils.logging import get_logger
from trident.utils.types.dataspec import EvaluationDict, PreprocessingDict
from trident.utils.types.dataset import DatasetProtocol

log = get_logger(__name__)


PREPROCESSING_KWDS = {"method", "apply"}


class TridentDataspec:
    r"""
    A class to handle data specification in trident.

    This class is designed to instantiate the dataset,
    preprocess the dataset, and create data loaders for training, validation, or testing.

    The preprocessing configuration includes two special keys:
    - 'method': Holds dictionaries of class methods and their keyword arguments for preprocessing.
    - 'apply': Contains dictionaries for user-defined functions and their keyword arguments to apply on the dataset.

    Attributes:
        name (str): Name of the dataspec, helpful for logging and tracking.
        cfg (DictConfig): The configuration object that contains all the settings for dataset
                          instantiation, preprocessing, dataloader setup, and evaluation metrics.

    The below example denotes a complete ``TridentDataspec`` configuration.
    The configuration is typically composed hierarchically, pulling in ``default.yaml`` for the indivdual
    keys.

    .. code-block:: yaml

        # Example TridentDataspec configuration for NLI dataset
        mnli_train:  # dataspec name passed over as `name`, required
          # below dict is passed as `cfg`
          misc: # optional, reserved for keywords that do not fit otherwise
            remove_unused_columns: true # if passed, boolean or null (`None` in Python)
          dataset: # required, config on how to instantiate dataset
            _target_: datasets.load_dataset
            path: glue
            name: mnli
            split: train
          preprocessing: # optional
            method:
              map:
                function:
                  _target_: src.tasks.text_classification.processing.preprocess_fn
                  _partial_: true
                  column_names:
                    text: premise
                    text_pair: hypothesis
                  tokenizer:
                    _partial_: true
                    _target_: transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__
                    self:
                        _target_: transformers.AutoTokenizer.from_pretrained
                        pretrained_model_name_or_path: ${module.model.pretrained_model_name_or_path}
                    padding: false
                    truncation: true
                    max_length: 128
          dataloader: # required
            _target_: torch.utils.data.dataloader.DataLoader
            collate_fn:
              _target_: tests.helpers.data.collate_fn
              _partial_: true
            batch_size: 10
            num_workers: 0
            pin_memory: true
          evaluation: # optional, for `val` and `test` keys in datamodule config
            # prepare fns all need to return original object
            prepare:
              # fn that takes (module: TridentModule, batch: dict, split: trident.utils.enums.Split)
              batch: null
              # fn takes (module: TridentModule, outputs: dict, batch: dict, split: trident.utils.enums.Split)
              outputs:
                _partial_: true
                _target_: src.tasks.text_classification.evaluation.get_preds
              # takes (module: TridentModule, flattened_step_outputs: dict, stage: str)
              # where list of step_outputs are flattened
              step_outputs: null
            # Which keys/attributes are supposed to be collected from `outputs` and `batch`
            step_outputs:
              outputs:
                - "preds" # can be a str
                - "logits"
              batch: # or a list[str]
                - labels
            # either metrics or val_metrics and test_metrics
            # where the latter
            # metrics should be copied for each dataset by default unless _datasets_ is specified
            metrics:
              # name of the metric used eg for logging
              acc:
                # instructions to instantiate metric, preferrably torchmetrics.Metric
                metric:
                  _partial_: true
                  _target_: torchmetrics.functional.accuracy
                # either "eval_step" or "epoch_end"
                compute_on: "epoch_end"
                kwargs:
                  preds: "outputs:preds"
                  target: "outputs:labels"
    """

    def __init__(self, cfg: DictConfig, name: str = "None"):
        r"""
        Initializes the TridentDataspec instance.

        Args:
            cfg: The Hydra configuration object for this dataspec.
            name: The name of the dataspec. Defaults to ``"None"``.
        """
        self.name = name
        self.cfg = cfg
        self.dataset: DatasetProtocol = self.preprocess(
            hydra.utils.instantiate(self.cfg.dataset), self.cfg.get("preprocessing")
        )
        if hasattr(self.cfg, "evaluation"):
            self.evaluation: EvaluationDict = hydra.utils.instantiate(
                self.cfg.evaluation
            )

    @staticmethod
    def preprocess(dataset: Any, cfg: Optional[PreprocessingDict]) -> Any:
        r"""
        Applies preprocessing steps to the dataset as specified in the config.

        The ``cfg`` includes two special keys:
        - ``"method"``: Holds dictionaries of class methods and their keyword arguments for preprocessing.
        - ``"apply"``: Contains dictionaries for user-defined functions and their keyword arguments to apply on the dataset.

        The preprocessing fucntions take the ``Dataset`` as the first positional argument. The functions are called in order of the configuration. Note that ``"method"`` is a convenience keyword which can also be achieved by pointing to the classmethod in ``"_target_"`` of an ``"apply"`` function.

        Args:
            dataset: The dataset to be preprocessed.
            cfg: A dictionary of preprocessing configurations.

        Returns:
            Any: The preprocessed dataset.
        """
        if cfg is None:
            return dataset
        cfg_ = deepcopy(cfg)
        extra_kwds = [k for k in cfg.keys() if k in PREPROCESSING_KWDS]
        for kwd in extra_kwds:
            kwd_cfg = cfg_[kwd]
            for key, key_cfg in kwd_cfg.items():
                # _method_ is for convenience
                # construct partial wrapper, instantiate with cfg, and apply to ret
                if kwd == "method":
                    key_cfg[
                        "_target_"
                    ] = f"{dataset.__class__.__module__}.{dataset.__class__.__name__}.{key}"
                # methods and functions should take dataset as first positional argument
                val = hydra.utils.instantiate(key_cfg, dataset)
                # `fn` might mutate ret in-place
                if val is not None:
                    dataset = val
        return dataset

    def get_dataloader(
        self, signature_columns: Optional[Sequence[str]] = None
    ) -> DataLoader:
        """
        Creates a DataLoader for the dataset.

        Args:
            signature_columns: Columns to be used in the dataloader. Defaults to None.
            If passed, removes unused columns if configured in ``misc``.

        Returns:
            DataLoader: The DataLoader configured as per the specified settings.
        """
        dataset: DatasetProtocol = self.dataset
        if signature_columns is not None:
            if (misc_cfg := self.cfg.get("misc")) and misc_cfg.get(
                "remove_unused_columns"
            ):
                dataset: DatasetProtocol = self._remove_unused_columns(
                    signature_columns
                )
        return hydra.utils.call(self.cfg.dataloader, dataset=dataset)

    def _remove_unused_columns(
        self, signature_columns: Sequence[str]
    ) -> DatasetProtocol:
        """
        Removes columns from the dataset that are not used.

        Args:
            signature_columns: A sequence of column names to keep in the dataset.

        Returns:
            Dataset: The modified dataset with unused columns removed.
        """
        column_names: None | list[str] = getattr(self.dataset, "column_names", None)
        if column_names is not None:
            signature_columns = (
                signature_columns if signature_columns is not None else []
            )
            ignored_columns = list(set(column_names) - set(signature_columns))
            if len(ignored_columns) > 0:
                log.info(
                    f"The following columns don't have a corresponding argument in "
                    f"`model forward` and have been ignored: {', '.join(ignored_columns)}."  # type: ignore
                )
            # ignoring as new_fingerprint typically not passed
            return self.dataset.remove_columns(ignored_columns)  # type: ignore
        else:
            dataset_name_ = self.name if isinstance(self.name, str) else "unnamed"
            log.warning(
                f"Attempting to remove unused columns for unsupported dataset {dataset_name_}!"
            )
            return self.dataset
