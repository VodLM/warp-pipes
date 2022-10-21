from __future__ import annotations

from enum import Enum
from functools import singledispatch
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import tensorstore as ts
import torch

from warp_pipes.support.shapes import infer_shape


TensorLike = Union[List, np.ndarray, torch.Tensor, ts.TensorStore]
TensorKey = Union[int, slice, List[int], np.ndarray]


class TensorFormat(Enum):
    NUMPY = "numpy"
    TORCH = "torch"


class TensorHanlder:
    """A utility class to handle different tensor formats."""

    def __init__(self, format: TensorFormat = TensorFormat.NUMPY) -> None:
        self.format = format

    def __call__(self, x: TensorLike, key: TensorKey = None) -> TensorLike:
        if key is not None:
            x = slice_tensor(x, key)
        x = format_tensor(x, self.format)
        return x


class TensorContrains:
    def __init__(self, types: Optional[List] = None, dim: Optional[int] = None) -> None:
        self.types = types
        self.dim = dim

    def __call__(
        self, x: TensorLike, explains: bool = False
    ) -> bool | Tuple[bool, str]:
        accepted = True
        explain_str = ""
        if self.types is not None:
            accepted_type = isinstance(x, tuple(self.types))
            if not accepted_type:
                explain_str += f"Type {type(x)} is not in {self.types}. "
                accepted &= accepted_type

        if self.dim is not None:
            shape = infer_shape(x)
            accepted_dim = len(shape) == self.dim
            if not accepted_dim:
                explain_str += (
                    f"Shape {shape} (dim={len(shape)}) is not of dimension {self.dim}. "
                )
            accepted &= accepted_dim

        if explains:
            return accepted, explain_str
        return accepted


@singledispatch
def format_tensor(x: TensorLike, format: TensorFormat) -> TensorLike:
    raise TypeError(f"Unsupported input type: {type(x)}")


@format_tensor.register(list)
def _(x: np.ndarray, format: TensorFormat) -> TensorLike:
    if format == TensorFormat.NUMPY:
        return np.array(x)
    elif format == TensorFormat.TORCH:
        return torch.tensor(x)
    else:
        raise ValueError(f"Unsupporteds format: {format}")


@format_tensor.register(np.ndarray)
def _(x: np.ndarray, format: TensorFormat) -> TensorLike:
    if format == TensorFormat.NUMPY:
        return x
    elif format == TensorFormat.TORCH:
        return torch.from_numpy(x)
    else:
        raise ValueError(f"Unsupporteds format: {format}")


@format_tensor.register(torch.Tensor)
def _(x: torch.Tensor, format: TensorFormat) -> TensorLike:
    if format == TensorFormat.NUMPY:
        x = x.detach().cpu()
        return x.numpy()
    elif format == TensorFormat.TORCH:
        return x
    else:
        raise ValueError(f"Unsupporteds format: {format}")


@format_tensor.register(ts.TensorStore)
def _(x: ts.TensorStore, format: TensorFormat) -> TensorLike:
    x = x.read().result()
    return format_tensor(x, format)


@singledispatch
def slice_tensor(x: TensorLike, key: TensorKey) -> TensorLike:
    return x[key]


@slice_tensor.register(ts.TensorStore)
def _(x: ts.TensorStore, key: TensorKey) -> ts.TensorStore:
    return x[key].read().result()


def tensor_constrains(
    *args_constrains: TensorContrains, **kwargs_constrains: TensorContrains
):
    def check_accepts(f):
        def new_f(*args, **kwds):
            for arg, constrains in zip(args, args_constrains):
                is_valid, explain = constrains(arg, explains=True)
                if not is_valid:
                    raise ValueError(explain)
            for key, constrains in kwargs_constrains.items():
                is_valid, explain = constrains(kwds[key], explains=True)
                if not is_valid:
                    raise ValueError(explain)
            return f(*args, **kwds)

        new_f.__name__ = f.__name__
        return new_f

    return check_accepts
