import functools
import io
from typing import Any

import numpy as np
import torch
import xxhash
from datasets.fingerprint import Hasher
from torch import nn


@functools.singledispatch
def get_fingerprint(obj: Any) -> str:
    """Fingerprint a Python object."""
    hash = Hasher()
    if isinstance(obj, torch.Tensor):
        obj = serialize_tensor(obj)
    hash.update(obj)
    return hash.hexdigest()


def serialize_tensor(x: torch.Tensor):
    """Convert a torch.Tensor into a bytes object."""
    x = x.cpu()
    buff = io.BytesIO()
    torch.save(x, buff)
    buff.seek(0)
    return buff.read()


@get_fingerprint.register(nn.Module)
def get_module_weights_fingerprint(obj: nn.Module) -> str:
    """Fingerprint a the weights of a PyTorch module."""
    hasher = xxhash.xxh64()
    state = obj.state_dict()
    for (k, v) in sorted(state.items(), key=lambda x: x[0]):
        hasher.update(k)
        u = serialize_tensor(v)
        hasher.update(u)

    return hasher.hexdigest()
