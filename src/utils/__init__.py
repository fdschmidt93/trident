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
            dim = v[0].dim()
            # stack matrices along first axis
            if dim >= 2:
                ret[k] = torch.vstack(v)
            # concatenate vectors
            elif dim == 1:
                ret[k] = torch.cat(v, dim=0)
            # create vectors from list of single tensors
            else:
                ret[k] = torch.stack(v)
        elif isinstance(v[0], np.ndarray):
            ret[k] = np.vstack(v)
        else:
            pass
    return ret
