from functools import lru_cache
from typing import Any, Mapping, Optional, Union, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from trident.utils.enums import Split
from trident.utils.logging import get_logger

EXTRA_KEYS = ["_method_", "_apply_"]
SPECIAL_KEYS = {"_datasets_"}

log = get_logger(__name__)


# DictConfig is hashable
@lru_cache
def get_dataset_cfg(
    cfg: Optional[Any],
    split: Optional[Split],
    dataset_name: Optional[str],
) -> Any:
    """Retrieve split config or value at lowest available level in hierarchy of `cfg`."""

    if not isinstance(cfg, Mapping):
        return cfg

    # Create a newly-typed reference to avoid type inference issues.
    current_cfg: Mapping = cfg

    # Retrieve split-specific configuration or default to base config.
    current_cfg = (
        current_cfg.get(split.value, current_cfg) if split is not None else current_cfg
    )
    current_cfg = current_cfg.get("_datasets_", current_cfg)
    current_cfg = (
        current_cfg.get(dataset_name, current_cfg) if dataset_name else current_cfg
    )

    return current_cfg


# TODO(fdschmidt93): test when wrapped up in partial
def get_local(var):
    import inspect

    frame = inspect.currentframe()
    locals: Union[None, dict] = getattr(getattr(frame, "f_back"), "f_locals")
    if locals is None:
        return None
    args = var.split(".")
    objects = [locals.get(args[0], None)]
    for i, arg in enumerate(args[1:]):
        val = getattr(objects[i], arg, None)
        objects.append(val)
    return objects[-1]


def expand(
    cfg: DictConfig, merge_keys: Union[str, list[str]], gen_keys: bool = False
) -> DictConfig:
    """Expands partial configuration of `keys` in `cfg` with the residual configuration.

    Most useful when configuring modules that have a substantial shared component.

    Applied by default on :obj:`datasets` (with :code:`gen_keys=False`) and :obj:`dataloaders` (with :code:`gen_keys=True`) of :obj:`DataModule` config.

    Notes:
        - Shared config reflects all configuration excluding set :obj:`keys`.

    Args:
        merge_keys (:obj:`Union[str, list[str])`):
            Keys that comprise dedicated configuration for which shared config will be merged.

        gen_keys (:obj:`bool`):
            Whether (:code:`True`) or not (:code:`False) to create :code:`keys` in :code:`cfg: with shared configuration if :code:`keys` do not exist yet.

    Example:
        :code:`expand(cfg.dataloaders, keys=["train", "val", "test"], gen_keys=True)` with the following config

        .. code-block:: yaml

            dataloaders:
              collate_fn:
                _target_: val_test_collator
              num_workers: 8
              shuffle: False
              train:
                collate_fn:
                  _target_: train_collator
                shuffle: true
                _datasets_:
                  source:
                    batch_size: 8
                  target:
                    batch_size: 16
              val:
                num_workers: 4
              test:
                num_workers: 2

        resolves to

        .. code-block:: yaml

            dataloaders:
              train:
                _datasets_:
                  source:
                    collate_fn:
                      _target_: train_collator
                    num_workers: 8
                    shuffle: true
                    batch_size: 8
                  target:
                    collate_fn:
                      _target_: train_collator
                    num_workers: 8
                    shuffle: true
                    batch_size: 16
              val:
                collate_fn:
                  _target_: val_test_collator
                num_workers: 4
                shuffle: false
              test:
                collate_fn:
                  _target_: val_test_collator
                num_workers: 2
                shuffle: false

        while only the original config is the one being logged.
    """

    if cfg is None:
        raise ValueError("Provided configuration (cfg) should not be None.")

    # Ensure merge_keys is a list
    merge_keys = [merge_keys] if isinstance(merge_keys, str) else merge_keys

    # Get the shared configuration keys
    shared_keys = [str(key) for key in cfg.keys() if key not in merge_keys]
    shared_cfg = OmegaConf.masked_copy(cfg, shared_keys)
    shared_cfg = cast(DictConfig, _merge_cfg(shared_cfg))

    # for merge keys ensure that all shared config is in special sub configs, merged top-down
    out_cfg = OmegaConf.create({})
    for key in merge_keys:
        key_cfg = cfg.get(key, OmegaConf.create({}) if gen_keys is True else None)
        if key_cfg is not None:
            # Avoids that top-level config within a key_cfg does not get
            # erroneously overriden by `shared_cfg`,
            # ie if it is not also defined in special key level of key_cfg
            for k in SPECIAL_KEYS:
                for sk in shared_cfg.get(k, []):
                    key_ = f"{k}.{sk}"
                    if not OmegaConf.select(key_cfg, key_):
                        OmegaConf.update(key_cfg, key_, OmegaConf.create({}))
            # we have to do an outer `_merge_cfg` since `shared_cfg` most likely
            # does not comprise `_Special_Keys`
            out_cfg[key] = _merge_cfg(
                cast(DictConfig, OmegaConf.merge(shared_cfg, _merge_cfg(key_cfg)))
            )
    return out_cfg


def _merge_cfg(cfg: DictConfig):
    """Merges top-level of `cfg` into sub-level special keys if they exist."""
    # early return in simple scenario
    if not any(k in cfg for k in SPECIAL_KEYS):
        return cfg

    # merge into new `out_cfg` that joins shared config into keys of special keys
    shared_keys = [str(k) for k in cfg.keys() if k not in SPECIAL_KEYS]
    shared_cfg = OmegaConf.masked_copy(cfg, shared_keys)
    out_cfg = OmegaConf.create({})
    for special_key in SPECIAL_KEYS:
        assert isinstance(
            cfg[special_key], DictConfig
        ), f"{cfg.special_key=} is not a DictConfig!"
        for sub_key in cfg[special_key].keys():
            if special_key not in out_cfg:
                out_cfg[special_key] = OmegaConf.create({})
            out_cfg[special_key][sub_key] = cast(
                DictConfig,
                OmegaConf.merge(shared_cfg, cfg[special_key][sub_key]),
            )
    return out_cfg


# TODO(fdschmidt93): update documentation once preprocessing routines are set
# TODO(fdschmidt93): add _keep_ to docs
def instantiate_and_apply(cfg: Union[None, DictConfig]) -> Any:
    r"""Adds :obj:`_method_` and :obj:`_apply_` keywords for :code:`hydra.utils.instantiate`.

    :obj:`_method_` and :obj:`_apply_` describe methods and custom functions to be applied on the instantiated object in order of the configuration. Most commonly, you want to make use of :obj:`dataset` `processing methods <https://huggingface.co/docs/datasets/process.html>`_\. For convenience

    Args:
        cfg (:obj:`omegaconf.dictconfig.DictConfig`):
            The :obj:`dataset_cfg` for your :obj:`TridentDataModule`

    Notes:
        - If :obj:`_method_` and :obj:`_apply_` are not set, :obj:`instantiate_and_apply` essentially reduces to :obj:`hydra.utils.instantiate`
        - :obj:`instantiate_and_apply` is only applied for the :obj:`dataset_cfg` of the TridentDataModule
        - The function arguments must be dictionaries
        - Any transformation is applied sequentially -- **order matters**!
            * If you want to intertwine :code:`_method_` and :code:`_apply_` use the former via the latter as per the final example.

    Example:
        The below example levers `datasets <https://huggingface.co/docs/datasets/>`_ to instantiate MNLI seamlessly with the below Python code and YAML configuration.

        .. code-block:: python

            dataset = instantiate_and_apply(cfg=dataset_cfg)

        .. code-block:: yaml

            dataset_cfg:
                _target_: datasets.load.load_dataset
                # required key!
                _recursive_: false
                # apply methods of _target_ object
                _method_:
                  map: # uses method of instantiated object, here dataset.map
                    # kwargs for dataset.map
                    function:
                      _target_: src.datamodules.preprocessing.{...}
                    batched: true
                  set_transform: # use dataset.set_transform
                    # kwargs for dataset.set_transform
                    _target_: src.custom.my_transform
                _apply_:
                    my_transform:
                        _target_: src.hydra.utils.partial
                        _partial_: src.custom.utils.my_transform

        The order of transformation is (1) :code:`map`, (2) :code:`set_transform`, and (3) :code:`my_transform`.

        \(1) :code:`map`, (2) :code:`my_transform`, and (3) :code:`set_transform` would be possible as follows.

        .. code-block:: yaml

            # ...
            _apply_:
                map:
                    _target_: dataset.arrow_dataset.Dataset.map
                    # ...
                my_transform:
                    _target_: src.hydra.utils.partial
                    _partial_: src.custom.utils.my_transform
                    # ...
                set_transform:
                    _target_: dataset.arrow_dataset.Dataset.set_transform
                    # ...

    Returns:
        Any: your instantiated object processed with _method_ & _apply_ functions
    """
    if cfg is None:
        return None

    # instantiate top-level cfg
    cfg_keys = list(cfg.keys())  # avoid changing dictionary size in loop
    extra_kwds = {key: cfg.pop(key) for key in cfg_keys if key in EXTRA_KEYS}
    obj = hydra.utils.instantiate(cfg)

    if not extra_kwds:
        return obj
    # kwd: {_method_, _apply_}
    # kwd_config: their respective collections of functions
    # key: name of user method or function
    # kwd_config: their respective config
    # TODO(fdschmidt93): handle ListConfig?
    for kwd, kwd_cfg in extra_kwds.items():
        for key, key_cfg in kwd_cfg.items():
            # _method_ is for convenience
            # construct partial wrapper, instantiate with cfg, and apply to ret
            if kwd == "_method_":
                key_cfg[
                    "_target_"
                ] = f"{obj.__class__.__module__}.{obj.__class__.__name__}.{key}"
                val = hydra.utils.instantiate(key_cfg, self=obj)
                # `fn` might mutate ret in-place
                if val is not None:
                    obj = val
            else:
                obj = key_cfg(obj)
    return obj


def config_callbacks(cfg: DictConfig, cb_cfg: DictConfig) -> DictConfig:
    """Amends configuration with user callback by configuration key.

    Hydra excels at depth-first, bottom-up config resolution. However,
    such a paradigm does not always allow you to elegantly express scenarios
    that are very relevant in experimentation. One instance, where :obj:`trident`
    levers :obj:`config_callback`s is the `Huggingface datasets <https://huggingface.co/docs/datasets/>`_ integration.

    An example configuration may look like follows:

    .. code-block:: yaml

        config: # global config
          datamodule:
            dataset_cfg:
              # ${SHARED}
              _target_: datasets.load.load_dataset
              #     trident-integration into huggingface datasets
              #     to lever dataset methods within yaml configuration
              _method_:
                function:
                  _target_: src.utils.hydra.partial
                  _partial_: src.datamodules.utils.preprocessing.text_classification
                  tokenizer:
                    _target_: src.datamodules.utils.tokenization.HydraTokenizer
                    pretrained_model_name_or_path: roberta-base
                    max_length: 53
                batched: false
                num_proc: 12

              path: glue
              name: mnli

              # ${INDIVIDUAL}
              train:
                split: "train"
                # ${SHARED} will be merged into {train, val test} with priority for existing config
              val:
                split: "validation_mismatched+validation_matched"
              test:
                path: xtreme # overrides shared glue
                name: xnli # overrides shared mnli
                lang: de
                split: "test"


    Args:
        cfg:
        cb_cfg:

    Returns:
        DictConfig:

    .. seealso:: :py:func:`src.utils.hydra.expand`, :py:func:`src.utils.hydra.instantiate_and_apply`, :py:func:`src.datamodule.utils.load_dataset`
    """
    for key in cb_cfg:
        if to_process_cfg := OmegaConf.select(cfg, str(key)):
            OmegaConf.resolve(to_process_cfg)
            processed_cfg = hydra.utils.call(cb_cfg.get(key), to_process_cfg)
            OmegaConf.update(cfg, str(key), processed_cfg)
        else:
            log.info(f"Attempted to mutate non-existing {key} configuration.")
    return cfg
