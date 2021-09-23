from typing import Optional, Tuple, Union

import torchmetrics
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf


def get_metrics(
    metrics: Optional[dict] = None,
) -> Union[dict[str, Tuple[str, torchmetrics.Metric]], None]:
    if isinstance(metrics, (dict, DictConfig, OmegaConf)):
        return {
            metric: (key, getattr(torchmetrics, metric)())
            for metric, key in metrics.items()
        }
    return None
