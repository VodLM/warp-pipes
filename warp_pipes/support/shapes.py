from numbers import Number
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import T
from typing import Tuple
from typing import Union

import numpy as np
from torch import Tensor

from warp_pipes.support.datastruct import Batch


class LeafType:
    """A utility class to represent the type of leaves in a nested list.
    Used to returns types in `infer_shape_nested_list_`.

    Warning: Hacky implementation, mostly for debugging purposes.
    """

    def __init__(self, leaves: List[Any]):
        self.types: List = []
        self.is_subtype = False
        self.lengths = []

        if all(isinstance(leaf, LeafType) for leaf in leaves):
            self.is_subtype = True

            if all(leaf.is_subtype for leaf in leaves):
                for leaf in leaves:
                    self.types += leaf.types
                    self.lengths += leaf.lengths
            else:
                self.is_subtype = True
                for leaf in leaves:
                    self.types += leaf.types
                    self.lengths += [len(leaf.types)]

            # if all lengths are equal, we don't consider it as a subtype since this
            # will appear as a dimension in the shape of the batch
            if len(set(self.lengths)) == 1:
                self.is_subtype = False
        else:
            for leaf in leaves:
                self.types.append(type(leaf))

    def __repr__(self):
        if self.is_subtype:
            if len(self.lengths):
                return (
                    f"{type(list())} {min(self.lengths)} to {max(self.lengths)} items, "
                    f"value_types={self._repr_types(self.types)}"
                )
            else:
                return f"Empty, value_types={self._repr_types(self.types)}"
        else:
            return self._repr_types(self.types)

    def _repr_types(self, types: List[type]):
        unique_types = set(types)
        return ", ".join(str(t) for t in unique_types)


def longest_sublist(lists: List[List[T]]) -> List[T]:
    """Returns the longest common sequence, starting from the start. """
    if len(lists) == 0:
        return []

    output = []
    ref = lists[0]
    for i in range(min(len(le) for le in lists)):
        x = ref[i]
        if all(le[i] == x for le in lists):
            output += [x]
        else:
            break
    return output


def infer_shape_nested_list_(x, level=0) -> Tuple[List[int], Optional[LeafType]]:
    """Recursive function used to infer the lengths of nested lists.
    See `infer_nested_lengths` for more details.
    The leaf level corresponds to the last level of the nested structure.
    if sub-fields are not of the same shape, this level is considered to be a leaf level.
    """
    sub_shapes = []
    leaf_types = []
    leaf_level = False

    if len(x) == 0:
        return [], LeafType(x)

    for y in x:
        if (
            isinstance(y, (list))
            or isinstance(y, (np.ndarray, Tensor))
            and len(y.shape) > 0
        ):
            sub_shape_y, leaf_type = infer_shape_nested_list_(y, level=level + 1)
            sub_shapes.append(sub_shape_y)
            leaf_types.append(leaf_type)
        else:
            leaf_level = True
            break

    if leaf_level:
        level_shape = [len(x)]
        level_leaf_type = LeafType(x)
    elif not all(sub_shapes[0] == s for s in sub_shapes):
        # in this case this is a leaf level, but we must transmit shared sub_shapes
        level_shape = [len(x), *longest_sublist(sub_shapes)]
        level_leaf_type = LeafType(leaf_types)
    else:
        level_shape = [len(sub_shapes), *sub_shapes[0]]
        level_leaf_type = LeafType(leaf_types)

    return level_shape, level_leaf_type


def infer_shape_nested_list(
    x: List[Any], return_leaf_type: bool = False
) -> Union[List[int], Tuple[List[int], LeafType]]:
    """Infer the lengths of nested lists along with the type of the leaves."""
    shape, leaf_type = infer_shape_nested_list_(x)
    if return_leaf_type:
        return shape, leaf_type
    else:
        return shape


def infer_shape(
    x: Union[List, np.ndarray, Tensor], return_leaf_type: bool = False
) -> Union[Optional[List[int]], Tuple[Optional[List[int]], str]]:
    """Infer shape of an array / nested lists"""
    if isinstance(x, Tensor):
        shape = list(x.shape)
        if return_leaf_type:
            _dtype = str(x.dtype).replace("torch.", "")
            return shape, f"{type(x)} (dtype={_dtype}, device={x.device})"
        else:
            return shape
    elif isinstance(x, np.ndarray):
        shape = list(x.shape)
        if return_leaf_type:
            return shape, f"{type(x)} (dtype={x.dtype})"
        else:
            return shape
    elif isinstance(x, (Number, str)) or x is None:
        if return_leaf_type:
            return [], str(type(x))
        else:
            return []
    elif isinstance(x, list):
        return infer_shape_nested_list(x, return_leaf_type=return_leaf_type)
    else:
        raise TypeError(f"Unsupported type {type(x)}")


def infer_batch_shape(
    batch: Batch, return_all_shapes: bool = False
) -> Union[List[int], Tuple[List[int], Dict[str, List[int]]]]:
    """Infer the batch shape, which is defined as the longest common shape of all the fields."""
    shapes = {k: infer_shape(b) for k, b in batch.items()}
    non_null_shapes = [s for s in shapes.values() if s is not None and len(s) > 0]
    shape = longest_sublist(non_null_shapes)

    if return_all_shapes:
        return shape, shapes
    else:
        return shape


def infer_missing_dims(n_elements: int, *, shape: List[int]) -> List[int]:
    """Infer the missing dimensions in a shape (i.e. negative values)."""
    assert all(y != 0 for y in shape)
    if all(y > 0 for y in shape):
        return shape
    else:
        neg_idx = [i for i, y in enumerate(shape) if y < 0]
        if len(neg_idx) != 1:
            raise ValueError(f"Only one dimension can be negative. Found: shape{shape}")
        neg_idx = neg_idx[0]
        known_dims = [y for y in shape if y > 0]

        p = np.prod(known_dims)
        if n_elements % p != 0:
            raise ValueError(
                f"n_elements={n_elements} must be divisible by product of known "
                f"dimensions, p={p} (Known dimensions: {known_dims}, shape={shape})"
            )
        missing_dim = n_elements // p
        shape[neg_idx] = missing_dim

    return shape


def infer_batch_size(batch: Batch) -> int:
    batch = {
        k: v
        for k, v in batch.items()
        if v is not None and not isinstance(v, (Number, str))
    }
    bss = [len(v) for v in batch.values()]
    bs = bss[0]
    if not all(bs == bs_ for bs_ in bss):
        lengths = ", ".join([f"{k}={len(v)}" for k, v in batch.items()])
        raise ValueError(
            f"Fields are not of the same length. Cannot infer batch size. Lengths=({lengths})"
        )
    return bs


def infer_stride(batch: Batch) -> int:
    def _cond(k):
        return not k.startswith("__") and not k.endswith("__")

    x = next(iter(v for k, v in batch.items() if _cond(k)))
    stride = len(next(iter(x)))
    assert all(stride == len(y) for y in x)
    return stride
