from typing import Optional, Tuple, Union

import torchmetrics
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf


class TorchMetric(torchmetrics.Metric):
    def __init__(self,
        val_metrics: Union[dict, DictConfig, OmegaConf],
        test_metrics: Union[dict, DictConfig, OmegaConf]) -> None:



def get_metrics(
    metrics: Optional[dict] = None,
) -> Union[dict[str, Tuple[str, torchmetrics.Metric]], None]:
    if isinstance(metrics, (dict, DictConfig, OmegaConf)):
        return {
            metric: (key, getattr(torchmetrics, metric)())
            for metric, key in metrics.items()
        }
    return None
