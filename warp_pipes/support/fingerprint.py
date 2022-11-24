from __future__ import annotations

import functools
import io
from numbers import Number
from typing import Any

import numpy as np
import torch
import xxhash
from datasets.fingerprint import Hasher
from tensorstore import TensorStore
from torch import nn

from warp_pipes.support.tensor_handler import TensorFormat
from warp_pipes.support.tensor_handler import TensorHandler


@functools.singledispatch
def get_fingerprint(obj: Any) -> str:
    """Fingerprint a Python object."""
    return _get_object_hash(obj)


def _get_object_hash(obj: Any) -> str:
    """Fingerprint a Python object."""
    hash = Hasher()
    if isinstance(obj, torch.Tensor):
        obj = serialize_tensor(obj)
    hash.update(obj)
    return hash.hexdigest()


@get_fingerprint.register(str)
@get_fingerprint.register(Number)
@get_fingerprint.register(type(None))
def get_displayable_fingerprint(x: str | Number | None) -> str:
    x = str(x)
    if len(x) <= 14:
        return f"<{x}>"
    else:
        return _get_object_hash(x)


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


@get_fingerprint.register(TensorStore)
@get_fingerprint.register(np.ndarray)
@get_fingerprint.register(torch.Tensor)
def get_tensor_fingerprint(
    x: TensorStore | np.ndarray | torch.tensor, *, _chunk_size: int = 1000
) -> str:
    hasher = xxhash.xxh64()
    handler = TensorHandler(format=TensorFormat.NUMPY)
    for i in range(0, x.shape[0], _chunk_size):
        chunk = handler(x, key=slice(i, i + _chunk_size))
        u = serialize_tensor(chunk)
        hasher.update(u)
    return hasher.hexdigest()


def serialize_tensor(x: torch.Tensor | np.ndarray) -> bytes:
    """Convert a torch.Tensor into a bytes object."""
    buff = io.BytesIO()
    if isinstance(x, torch.Tensor):
        if x.is_sparse:
            x = x.to_dense()
        x = x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        pass
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

    np.savez(buff, x)
    buff.seek(0)
    return buff.read()
