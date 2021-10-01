from dataclasses import dataclass
from typing import Optional

from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import MISSING


@dataclass
class BaseDataModuleConfig:
    collate_fn: DictConfig = MISSING
    batch_size: int = 8
    num_workers: int = 8
    pin_memory: bool = True
    seed: int = 42


@dataclass
class MetaDataModuleConfig:
    train_dm: BaseDataModuleConfig = BaseDataModuleConfig()
    val_dm: BaseDataModuleConfig = BaseDataModuleConfig()
    test_dm: BaseDataModuleConfig = BaseDataModuleConfig()


@dataclass
class OptimizerConfig:
    """
    Config spec for the optimizer.
    Args:
        **kwargs: 
            _target_: str: import path of optimizer to instantiate 
    
    Other **kwargs result from the corresponding optimizer to be instantiated.
            
    Returns: hydra.utils.instantiate(OptimizerConfig, optimizer)
    """

    ...


@dataclass
class SchedulerConfig:
    """
    Config spec for the optimizer scheduler.
    Args:
        **kwargs: 
            _target_: str: import path of scheduler to instantiate
            num_warmup_steps: Union[int, float] = total or relative number of (total) steps
            num_training_steps: int = total number of training steps
    
    Notes:
        - num_training_steps is automatically added and computed on-the-fly
            
    Returns: hydra.utils.instantiate(SchedulerConfig, optimizer)
    """

    ...

@dataclass
class MetricsConfig:
    """
    Config spec for an evaluation metric.
    Args:
        **kwargs: 
            metric:
                _target_: str: a torchmetrics.Metric or subclass thereof
            num_warmup_steps: Union[int, float] = total or relative number of (total) steps
            num_training_steps: int = total number of training steps
    
    Notes:
        - num_training_steps is automatically added and computed on-the-fly

    Example:
        ```yaml
        metrics:
          accuracy:                             # name
            metric:                             # metric to be computed
              _target_: torchmetrics.Accuracy   # pointer
            batch: "labels"                     # attr of batch to target
            outputs: "preds"                    # attr of outputs to preds
        ```
            
    Returns: hydra.utils.instantiate(SchedulerConfig, optimizer)
    """

    ...

@dataclass
class EvaluationConfig:
    """
    Config spec for the optimizer scheduler.
    Args:
        **kwargs: 
            _target_: str: import path of scheduler to instantiate
            num_warmup_steps: Union[int, float] = total or relative number of (total) steps
            num_training_steps: int = total number of training steps
    
    Notes:
        - num_training_steps is automatically added and computed on-the-fly
            
    Returns: hydra.utils.instantiate(SchedulerConfig, optimizer)
    """

    ...


@dataclass
class ModelConfig:
    """
    Config spec for the model class.



    """

    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: Optional[SchedulerConfig] = SchedulerConfig()


@dataclass
class Config:
    ...
