from __future__ import annotations

import math
from copy import copy
from copy import deepcopy
from functools import partial
from random import randint
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from warp_pipes.support.functional import always_true
from warp_pipes.support.json_struct import apply_to_json_struct
from warp_pipes.support.json_struct import flatten_json_struct
from warp_pipes.support.tensor_handler import tensor_constrains
from warp_pipes.support.tensor_handler import TensorContrains
from warp_pipes.support.tensor_handler import TensorFormat
from warp_pipes.support.tensor_handler import TensorHandler
from warp_pipes.support.tensor_handler import TensorLike

FillValue = TypeVar("FillValue", float, int, np.ndarray)
FLOAT_INF: float = 1e-18


def is_negative(x):
    return x < 0


@tensor_constrains(TensorContrains(dim=1))
def pad_to_length(
    lst: List[FillValue], *, length: int, fill_token: FillValue
) -> List[FillValue]:
    if len(lst) < length:
        if isinstance(lst, list):
            lst.extend([fill_token] * (length - len(lst)))
        elif isinstance(lst, np.ndarray):
            lst = np.pad(
                lst, (0, length - len(lst)), mode="constant", constant_values=fill_token
            )
        elif isinstance(lst, Tensor):
            lst = F.pad(lst, (0, length - len(lst)), value=fill_token)
        else:
            raise TypeError(f"Unsupported type: {type(lst)}")
    return lst[:length]


@tensor_constrains(TensorContrains(dim=2))
def pad_second_dim(arr: TensorLike, *, k: int, fill_token: Any) -> TensorLike:
    """Pad second dimension to length k."""

    if isinstance(arr, list):
        pad_fn = partial(pad_to_length, fill_token=fill_token, length=k)
        return list(map(pad_fn, arr))
    elif isinstance(arr, np.ndarray):
        if arr.shape[1] == k:
            return arr
        return np.pad(
            arr,
            ((0, 0), (0, k - arr.shape[1])),
            mode="constant",
            constant_values=fill_token,
        )
    elif isinstance(arr, Tensor):
        if arr.shape[1] == k:
            return arr
        return F.pad(arr, (0, k - arr.shape[1]), value=fill_token)
    else:
        raise TypeError(f"Unsupported type: {type(arr)}")


@tensor_constrains(TensorContrains(dim=2))
def unique_second_dim(arr: TensorLike, length: int, fill_token: Any = -1) -> TensorLike:
    """
    Remove duplicate rows in a 2d array.
    """
    pad_fn = partial(pad_to_length, fill_token=fill_token, length=length)
    if isinstance(arr, list):
        return [pad_fn(list(set(row))) for row in arr]
    elif isinstance(arr, np.ndarray):
        return np.stack([pad_fn(np.unique(a)) for a in arr])
    elif isinstance(arr, Tensor):
        return torch.stack([pad_fn(torch.unique(a)) for a in arr])
    else:
        raise TypeError(f"Unsupported type: {type(arr)}")


@tensor_constrains(TensorContrains(dim=2))
def masked_fill(
    arr: TensorLike,
    *,
    new_value: float | Tuple[int, int],
    condition: Optional[Callable] = None,
) -> TensorLike:
    """
    Replace all occurrences of replace_token in arr with new_token.
    """
    assert isinstance(new_value, (float, int, tuple))
    if condition is None:
        condition = always_true

    if isinstance(arr, list):
        has_neg_indices = any(condition(i) for i in flatten_json_struct(arr))
        if has_neg_indices:

            def _replace(x):
                if condition(x):
                    if isinstance(new_value, tuple):
                        return randint(*new_value)
                    else:
                        return new_value
                else:
                    return x

            return apply_to_json_struct(arr, _replace)
        else:
            return arr

    elif isinstance(arr, np.ndarray):
        if isinstance(new_value, tuple):
            rdn = np.random.randint(low=new_value[0], high=new_value[1], size=arr.shape)
        else:
            rdn = np.full_like(arr, new_value)
        return np.where(condition(arr), rdn, arr)
    elif isinstance(arr, Tensor):
        if isinstance(new_value, tuple):
            rdn = torch.randint_like(arr, low=new_value[0], high=new_value[1])
        else:
            rdn = torch.full_like(arr, new_value)
        return torch.where(condition(arr), rdn, arr)
    else:
        raise TypeError(f"Unsupported type: {type(arr)}")


def concat_tensors(*a: TensorLike, dim=0) -> TensorLike:
    """Concatenate tensors along a given dimension."""
    arr_type = type(a[0])
    assert all(isinstance(x, arr_type) for x in a)
    if arr_type == np.ndarray:
        return np.concatenate(a, axis=dim)
    elif arr_type == Tensor:
        return torch.cat(a, dim=dim)
    else:
        raise TypeError(f"Unsupported type: {type(a[0])}")


class SearchResult:
    """
    A small class to handle a batch of search results with `indices` and `scores`.
    When not enough results are returned, fill with negative indiceses.
    """

    def __init__(
        self,
        *,
        indices: TensorLike,
        scores: TensorLike,
        format: Optional[TensorFormat] = TensorFormat.TORCH,
    ):
        # format inputs and register attributes
        self.format = format
        formatter = TensorHandler(format)
        self.scores = formatter(scores)
        self.indices = formatter(indices)

        self.scores = formatter(self.scores)
        self.indices = formatter(self.indices)

        # check shapes
        assert self.scores.shape == self.indices.shape
        assert len(self.scores.shape) == 2

    def __repr__(self):
        scores_shape = self.scores.shape
        indices_shape = self.indices.shape
        return (
            f"{type(self).__name__}(scores={scores_shape}, "
            f"indices={indices_shape}, "
            f"format={self.format})"
        )

    def to(self, format: Optional[TensorFormat] = None) -> "SearchResult":
        formatter = TensorHandler(format)
        self.indices = formatter(self.indices)
        self.scores = formatter(self.scores)
        return self

    def copy(self, **new_attrs) -> "SearchResult":
        new_instance = copy(self)
        for k, v in new_attrs.items():
            setattr(new_instance, k, v)
        return new_instance

    def __add__(self, other: "SearchResult") -> "SearchResult":
        """
        Merge two search results. Keep unique indices and sum the scores. The mean scores
        is not known (and is probably not zero), therefore we offset the scores by the
        minimum scores from both inputs.
        """

        if not isinstance(other, SearchResult):
            raise TypeError(f"Unsupported type: {type(other)}")
        if len(self) != len(other):
            raise ValueError(
                f"Search results must have the same length, but {len(self)} != {len(other)}"
            )

        # convert to torch
        self = self.to(format=TensorFormat.TORCH)
        other = other.to(format=TensorFormat.TORCH)

        # take the minimum scores (except for `inf` values) and offset the scores
        min_scores_self = self._get_real_min(self.scores)
        self.scores = self.scores - min_scores_self[:, None]
        min_scores_other = self._get_real_min(other.scores)
        other.scores = other.scores - min_scores_other[:, None]

        # sum the scores based on the indices
        indices, scores = sum_scores(
            (self.indices, self.scores), (other.indices, other.scores)
        )

        # offset back the scores with the minimum scores
        scores += (min_scores_self + min_scores_other)[:, None]

        return self.copy(indices=indices, scores=scores).to(self.format)

    @staticmethod
    def _get_real_min(scores: Tensor) -> Tensor:
        scores_no_inf = torch.where(scores == -float("inf"), -scores, scores)
        min_scores_self = scores_no_inf.min(dim=1).values
        min_scores_self[torch.isinf(min_scores_self)] = 0.0
        return min_scores_self

    def __len__(self):
        return len(self.indices)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.indices.shape

    def resize(self, new_size: int) -> "SearchResult":
        if new_size <= self.shape[1]:
            new_indices = self.indices[:, :new_size]
            new_scores = self.scores[:, :new_size]
            return self.copy(indices=new_indices, scores=new_scores)
        else:
            new_indices = pad_second_dim(self.indices, k=new_size, fill_token=-1)
            new_scores = pad_second_dim(
                self.scores, k=new_size, fill_token=-float("inf")
            )
            return self.copy(indices=new_indices, scores=new_scores)

    def fill_masked_indices(self, value_range: Tuple[int, int]) -> "SearchResult":
        """Fill masked indices with random values in the range `value_range`."""
        new_indices = TensorHandler(format=TensorFormat.TORCH)(self.indices)
        new_indices = torch.where(
            new_indices < 0, torch.randint_like(new_indices, *value_range), new_indices
        )
        return self.copy(indices=new_indices).to(format=self.format)


def sum_scores(
    a: Tuple[Tensor, Tensor], b: Tuple[Tensor, Tensor], fill_token_value=-1
) -> Tuple[Tensor, Tensor]:
    """
    Sum the scores of two search results.

    Args:
        a (:obj:`Tuple[Tensor, Tensor]`): The first search result (indices, scores)
        b (:obj:`Tuple[Tensor, Tensor]`): The second search result (indices, scores)

    Returns:
        :obj:`Tuple[Tensor, Tensor]`: The sum of the two search results (indices, scores)
    """

    # unpack attributes
    a_indices, a_scores = a
    assert a_indices.shape == a_scores.shape
    b_indices, b_scores = b
    assert b_indices.shape == b_scores.shape

    # infer the new indices
    all_indices = torch.cat([a_indices, b_indices], dim=1)
    unique_indices, unique_inv = zip(
        *[torch.unique(t, return_inverse=True) for t in torch.unbind(all_indices)]
    )
    new_size = max(len(t) for t in unique_indices)
    unique_inv = torch.stack(unique_inv, dim=0)

    # set the new indices
    new_indices = fill_token_value + torch.zeros(
        (len(all_indices), new_size), dtype=torch.long
    )
    new_indices.scatter_(1, unique_inv, all_indices)

    # set the new scores
    all_scores = torch.cat([a_scores, b_scores], dim=1)
    new_scores = torch.zeros_like(new_indices, dtype=a_scores.dtype)
    new_scores.scatter_add_(1, unique_inv, all_scores)
    new_scores[new_indices == -1] = -math.inf

    # sort the indices and scores
    idx = torch.argsort(-new_scores, dim=1)
    new_scores = torch.gather(new_scores, 1, idx)
    new_indices = torch.gather(new_indices, 1, idx)

    return new_indices, new_scores
