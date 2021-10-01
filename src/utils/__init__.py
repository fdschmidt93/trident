from collections import defaultdict

import numpy as np
import torch


def flatten_dict(inputs: list[dict]) -> dict:
    """Conflates keys of list[dict] and stacks np arrays & tensors along 0-dim."""
    ret = defaultdict(list)
    for input_ in inputs:
        for k, v in input_.items():
            ret[k].append(v)
    for k, v in ret.items():
        if isinstance(v[0], torch.Tensor):
            ret[k] = torch.cat(v, dim=0)
        elif isinstance(v[0], np.ndarray):
            ret[k] = np.vstack(v)
        else:
            pass
    return ret
