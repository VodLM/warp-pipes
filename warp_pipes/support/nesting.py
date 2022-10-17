from functools import partial
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import T
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
import torch
from torch import Tensor

from warp_pipes.support.shapes import infer_missing_dims
from warp_pipes.support.shapes import infer_shape


def flatten_nested_(values: List[List], level=1, current_level=0) -> Iterable[Any]:
    """Flatten a nested list of lists. See `flatten_nested` for more details.

    Args:
      values (:obj:`List[List]`): The nested list to flatten.
      level (:obj:`int`, `optional`, defaults to 1): The level of nesting to flatten.
      current_level (:obj:`int`, `optional`, defaults to 0): The current level of
        nesting (don't set this manually).

    Returns:
        :obj:`Iterable[Any]`: The flattened list.
    """
    for x in values:
        if isinstance(x, (list, np.ndarray, Tensor)) and current_level < level:
            for y in flatten_nested_(x, level, current_level + 1):
                yield y
        else:
            yield x


def flatten_nested(values: Union[np.ndarray, Tensor, List[List]], level=1) -> List[Any]:
    """Flatten a nested array or list of lists up to level `level`.

    Args:
        values  (:obj:`Union[np.ndarray, Tensor, List[List]]`): The nested array or list to flatten.
        level (:obj:`int`, `optional`, defaults to 1): The level of nesting to flatten.

    Returns:
        :obj:`List[Any]`: The input array flattened up level `level`.
    """
    if isinstance(values, Tensor):
        return values.view(-1, *values.shape[level + 1 :])
    elif isinstance(values, np.ndarray):
        return values.reshape(-1, *values.shape[level + 1 :])
    elif isinstance(values, list):
        return list(flatten_nested_(values, level=level))
    else:
        raise TypeError(f"Unsupported type: {type(values)}")


def nested_list(
    values: List[Any], *, shape: Union[Tuple[int], List[int]], level=0
) -> List[Any]:
    """Nest a list of values according to `shape`. This functions is similar to `np.reshape`.

    Args:
        values (:obj:`List[Any]`): The list of values to nest.
        shape (:obj:`Union[Tuple[int], List[int]]`): The shape to nest the values into.
        level (:obj:`int`, `optional`, defaults to 0): The level of nesting to flatten.

    Returns:
        :obj:`List[Any]`: The input array nested up to level `level`.
    """
    if not isinstance(shape, list):
        shape = list(shape)
    shape = infer_missing_dims(len(values), shape=shape)
    stride = int(np.prod(shape[1:]))
    output = []
    for j in range(0, len(values), stride):
        section = values[j : j + stride]
        if len(shape) > 2:
            output += [nested_list(section, shape=shape[1:], level=level + 1)]
        else:
            output += [section]

    return output


def reconcat(values: List[Any], original_type: Type):
    if original_type == Tensor:
        values = torch.cat([t[None] for t in values], dim=0)
    elif original_type == np.ndarray:
        values = np.concatenate([t[None] for t in values], axis=0)
    elif original_type == list:
        pass
    else:
        raise ValueError(f"Cannot reconstruct values of original type={original_type}")
    return values


def expand_and_repeat(x: T, axis: int, n: int = 1) -> T:
    """Expand the axis and repeat `n` times"""
    shape = infer_shape(x)
    if axis < 0:
        axis = len(shape)

    if isinstance(x, np.ndarray):
        x = np.expand_dims(x, axis=axis)
        return np.repeat(x, n, axis=axis)

    elif isinstance(x, torch.Tensor):
        shape = x.shape
        x = x.unsqueeze(axis)
        a, b = shape[:axis], shape[axis:]
        new_shape = a + (n,) + b
        return x.expand(new_shape)

    elif isinstance(x, list):

        def repeat(x, *, n):
            return [x] * n

        def apply_at_level(
            fn: Callable, x: Iterable, *, level: int, current_level: int = 0
        ):
            if level == current_level:
                return fn(x)
            else:
                return [
                    apply_at_level(fn, y, level=level, current_level=current_level + 1)
                    for y in x
                ]

        return apply_at_level(partial(repeat, n=n), x, level=axis)


def nest_idx(idx: List[int], shape: List[int]) -> List[int]:
    """Get the index of a nested list"""
    stride = int(np.prod(shape[1:]))
    return [i * stride + j for i in idx for j in range(stride)]
