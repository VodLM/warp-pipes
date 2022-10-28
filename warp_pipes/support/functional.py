from __future__ import annotations

import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import T

import numpy as np
import torch

from warp_pipes.support.datastruct import Batch
from warp_pipes.support.datastruct import Eg
from warp_pipes.support.shapes import infer_batch_size


def indentity(x: T, **kwargs) -> T:
    return x


def always_true(*args, **kwargs):
    return True


def get_batch_eg(
    batch: Batch, idx: int | slice, filter_op: Optional[Callable] = None
) -> Dict:
    """Extract example `idx` from a batch, potentially filter the keys"""
    filter_op = filter_op or always_true
    return {k: v[idx] for k, v in batch.items() if filter_op(k)}


def iter_batch_egs(batch: Batch) -> Iterable[Eg]:
    batch_size = infer_batch_size(batch)
    for i in range(batch_size):
        yield get_batch_eg(batch, idx=i)


def camel_to_snake(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def cast_to_numpy(
    x: Any, as_contiguous: bool = True, dtype: Optional[str] = None
) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().to(device="cpu").numpy()
        if dtype is not None:
            x = x.astype(dtype)
    elif isinstance(x, np.ndarray):
        pass
    else:
        x = np.array(x)

    if as_contiguous:
        x = np.ascontiguousarray(x)

    return x


def check_equal_arrays(x, y):
    """check if x==y"""
    x = cast_to_numpy(x)
    y = cast_to_numpy(y)
    r = x == y
    if isinstance(r, np.ndarray):
        return r.all()
    elif isinstance(r, bool):
        return r
    else:
        raise TypeError(f"Cannot check equality of {type(r)}")
