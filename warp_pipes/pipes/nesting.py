from __future__ import annotations

import math
from functools import partial
from typing import Callable
from typing import List
from typing import Optional
from typing import T

import numpy as np
from datasets import Dataset
from datasets import DatasetDict
from torch import Tensor

from warp_pipes.core.condition import Condition
from warp_pipes.core.pipe import Pipe
from warp_pipes.pipes.basics import ApplyToAll
from warp_pipes.pipes.basics import Identity
from warp_pipes.support.datasets_utils import concatenate_datasets
from warp_pipes.support.datasets_utils import get_column_names
from warp_pipes.support.datasets_utils import HfDataset
from warp_pipes.support.datasets_utils import keep_only_columns
from warp_pipes.support.datasets_utils import remove_columns
from warp_pipes.support.datastruct import Batch
from warp_pipes.support.nesting import expand_and_repeat
from warp_pipes.support.nesting import flatten_nested
from warp_pipes.support.nesting import nest_idx
from warp_pipes.support.nesting import nested_list
from warp_pipes.support.nesting import reconcat
from warp_pipes.support.pretty import pprint_batch
from warp_pipes.support.pretty import repr_batch
from warp_pipes.support.shapes import infer_batch_shape
from warp_pipes.support.shapes import infer_batch_size
from warp_pipes.support.shapes import infer_nesting_level


def fshp(shp):
    return str(shp).replace("]", ", ...]")


class Flatten(Pipe):
    """Flatten a nested batch up to dimension=`level`.
    For instance a batch of shape (x, 3, 4, ...) with level=2 will be flattened to (x *3 * 4, ...)
    """

    def __init__(self, level: Optional[int] = 1, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.level = level

    def _call_batch(
        self,
        batch: Batch,
        idx: Optional[List[int]] = None,
        level: Optional[int] = None,
        **kwargs,
    ) -> Batch:
        level_ = self.level if level is None else level
        return {k: flatten_nested(v, level=level_) for k, v in batch.items()}

    @classmethod
    def instantiate_test(cls, **kwargs) -> None:
        return cls(level=1, **kwargs)


class Nest(ApplyToAll):
    """Nest a flat batch. This is equivalent to calling np.reshape to all values,
    except that this method can handle np.ndarray, Tensors and lists.
    If the target shape is unknown at initialization time, the `shape` attributed
    can be passed as a keyword argument to the __call__ method.
    """

    def __init__(self, shape: Optional[List[int]], **kwargs):
        """
        Args:
            shape (:obj:`List[int]`): The target shape. If None, the shape will be inferred
        """
        nest_fn = partial(self.nest, _shape=shape)
        super(Nest, self).__init__(
            element_wise=False, op=nest_fn, allow_kwargs=True, **kwargs
        )

    @staticmethod
    def nest(
        x: T,
        *,
        _shape: Optional[List[int]],
        shape: Optional[List[int]] = None,
        **kwargs,
    ) -> T:
        """Nest the input x according to shape or _shape.
        This allows specifying a shape that is not known at init.

        Args:
          x (:obj:`T`): The input to nest.
          shape (:obj:`List[int]`): Primary and optional target shape of the nested batch.
          _shape (:obj:`List[int]`, `optional`): Secondary and optional target shape of
            the nested batch, can be provided at runtime

        """
        shape = shape or _shape
        if shape is None:
            raise ValueError("Either shape or _shape must be provided")

        if isinstance(x, Tensor):
            return x.view(*shape, *x.shape[1:])
        elif isinstance(x, np.ndarray):
            return x.reshape((*shape, *x.shape[1:]))
        elif isinstance(x, list):
            return nested_list(x, shape=shape)
        else:
            raise TypeError(f"Unsupported type: {type(x)}")

    @classmethod
    def instantiate_test(cls, **kwargs) -> None:
        return cls([-1, 1], **kwargs)


class ApplyAsFlatten(Pipe):
    """Flattens the first `level+1` batch dimensions and
    applies the pipe to the flattened batch.

    Warning: Do not use this pipe if the inner pipe drops nested values
    or modifies the order of the batch elements!

    NB: This pipe is equivalent to:

    ```python
    # example data
    h = (20, 10) # some vector dimension
    nested_shape = (10, 8, 8) # some nested batch dimension
    batch = np.random.randn(size=([nested_shape, *h)]

    # ApplyAsFlatten(pipe)
    batch = batch.reshape(-1, *h)
    batch = pipe(batch)
    batch = batch.reshape(*nested_shape, *h)
    ```
    """

    flatten: Optional[Flatten] = None
    nest: Optional[Nest] = None

    def __init__(
        self,
        pipe: Pipe,
        level: int | List[str] = 1,
        flatten_idx: bool = True,
        flatten_as_dataset: bool = False,
        input_filter: Optional[Condition] = None,
        level_offset: int = 0,
        pprint: bool = False,
        **kwargs,
    ):
        if input_filter is None:
            input_filter = pipe.input_filter

        super(ApplyAsFlatten, self).__init__(input_filter=input_filter, **kwargs)
        self.pipe = pipe
        self.pprint = pprint
        self.level = level
        self.level_offset = level_offset
        self.flatten_idx = flatten_idx
        self.flatten_as_dataset = flatten_as_dataset
        self._skip_flatten = (level == 0) and not isinstance(level, list)
        if not self._skip_flatten:
            self.flatten = Flatten(level=None)
            self.nest = Nest(shape=None)
        else:
            self.flatten = self.nest = None

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        if self._skip_flatten:
            return self.pipe._call_batch(batch, **kwargs)

        # infer the flattening level
        level = self._infer_flattening_level(batch)
        pprint_batch(
            batch,
            f"ApplyAsFlatten::input (level={level}, "
            f"input_filter={self.pipe.input_filter})",
            silent=not self.pprint,
        )

        # infer the original shape of the batch
        ref_shape = infer_batch_shape(batch)[: level + 1]

        # flatten the batch
        if level > 0 and self.flatten is not None:
            batch = self.flatten(batch, level=level)

        pprint_batch(
            batch, f"ApplyAsFlatten::flattened (level={level})", silent=not self.pprint
        )

        # compute the new index
        if self.flatten_idx:
            idx = kwargs.get("idx", None)
            if idx is not None:
                kwargs = kwargs.copy()
                idx = nest_idx(idx, ref_shape)
                kwargs["idx"] = idx

        # apply the batch to the flattened batch
        batch = self.pipe(batch, **kwargs)

        pprint_batch(
            batch, f"ApplyAsFlatten::pipe::out (level={level})", silent=not self.pprint
        )

        # reshape back to the ref_shape
        if level > 0 and self.flatten is not None:
            output = self.nest(batch, shape=ref_shape)
        else:
            output = batch

        pprint_batch(
            batch,
            f"ApplyAsFlatten::pipe::nested (level={level})",
            silent=not self.pprint,
        )

        # check output and return
        new_shape = infer_batch_shape(output)
        new_shape = new_shape[: level + 1]
        explain = (
            "Applying a pipe that changes the batch size might have caused this issue."
        )
        if new_shape != ref_shape:
            raise ValueError(
                f"{new_shape} != {ref_shape}. Level={level}. "
                f"{explain}\n"
                f"{repr_batch(batch, header='ApplyAsFlatten output batch')}"
            )
        return output

    def _infer_flattening_level(self, batch):
        if isinstance(self.level, int):
            level = self.level
        elif isinstance(self.level, list):
            if not set.intersection(set(self.level), set(batch.keys())):
                raise ValueError(
                    f"Reference columns `{self.level}` not found in batch "
                    f"with keys: {batch.keys()}"
                )
            ref_batch = {k: v for k, v in batch.items() if k in self.level}
            level = infer_nesting_level(ref_batch)
        else:
            raise TypeError(f"Unsupported type for level: {type(self.level)}")

        level += self.level_offset
        return max(0, level)

    def _call_dataset_dict(self, dataset: DatasetDict, **kwargs) -> DatasetDict:
        return self._call_dataset_any(dataset, **kwargs)

    def _call_dataset(
        self,
        dataset: Dataset,
        **kwargs,
    ) -> Dataset:
        return self._call_dataset_any(dataset, **kwargs)

    def _call_dataset_any(self, dataset: HfDataset, **kwargs) -> HfDataset:
        if not self.flatten_as_dataset or self._skip_flatten:
            if isinstance(dataset, Dataset):
                return super()._call_dataset(dataset, **kwargs)
            elif isinstance(dataset, DatasetDict):
                return super()._call_dataset_dict(dataset, **kwargs)
            else:
                raise TypeError(f"Unsupported type: {type(dataset)}")
        else:
            # filter the dataset
            if self.input_filter is not None:
                new_dataset = keep_only_columns(dataset, self.input_filter)
            else:
                new_dataset = dataset

            desc = kwargs.pop("desc", type(self).__name__)
            batch_size = kwargs.pop("batch_size", 10)
            batch_eg = self._get_batch_example(batch_size, new_dataset)
            level = self._infer_flattening_level(batch_eg)
            input_full_shape = infer_batch_shape(batch_eg)
            input_full_shape[0] = -1
            ref_shape = input_full_shape[: level + 1]
            flatten_batch_size = batch_size * math.prod(ref_shape[1:])
            flat_shape = [-1, *input_full_shape[len(ref_shape) :]]
            if level > 0:
                new_dataset = self.flatten(
                    new_dataset,
                    **kwargs,
                    level=level,
                    batch_size=batch_size,
                    desc=f"{desc}: {fshp(input_full_shape)} -> {fshp(flat_shape)}",
                )

            # transform the dataset
            new_dataset = self.pipe(
                new_dataset,
                **kwargs,
                batch_size=flatten_batch_size,
                desc=f"{desc}",
            )

            # re-shape
            # NB: `num_proc=1` : avoid splitting a single example across multiple workers
            kwargs = {**kwargs, "num_proc": 1}
            if level > 0:
                new_dataset = self.nest(
                    new_dataset,
                    **kwargs,
                    batch_size=flatten_batch_size,
                    desc=f"{desc}: {fshp(flat_shape)} -> {fshp(input_full_shape)}",
                    shape=ref_shape,
                )

            # update the dataset
            if self.update:
                cols = get_column_names(dataset)
                cols_to_remove = [c for c in get_column_names(new_dataset) if c in cols]
                missing_columns = remove_columns(dataset, cols_to_remove)
                new_dataset = concatenate_datasets(
                    [missing_columns, new_dataset], axis=1
                )

            return new_dataset

    def _get_batch_example(self, batch_size, dataset):
        if isinstance(dataset, DatasetDict):
            dataset = next(iter(dataset.values()))
        batch_eg = dataset[:batch_size]
        return batch_eg

    @classmethod
    def instantiate_test(cls, **kwargs) -> None:
        return cls(Identity(), **kwargs)


class NestedLevel1(Pipe):
    """Apply a pipe to each nested value, handling each nested field as a separate batch.
    This can be use to modify the nested field inplace  (i.e. sorting, deleting).
    However the all pipe output must have the same batch size.
    """

    def __init__(self, pipe: Pipe, **kwargs):
        super().__init__(**kwargs)
        self.pipe = pipe

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        # get initial parameters
        keys = list(batch.keys())
        batch_size = infer_batch_size(batch)
        types = {k: type(v) for k, v in batch.items()}

        # process each Eg separately
        egs = (self.get_eg(batch, idx=i) for i in range(batch_size))
        egs = [self.pipe(eg, **kwargs) for eg in egs]

        # update `keys` with keys that were added by the pipe
        for key in egs[0].keys():
            if key not in keys:
                keys.append(key)
                types[key] = type(egs[0][key])

        # check shape consistency before re-concatenating
        batch_sizes = [infer_batch_size(eg) for eg in egs]
        bs = batch_sizes[0]
        if not all(bs == b for b in batch_sizes):
            raise ValueError(
                f"Batch sizes are inconsistent. "
                f"Make sure the pipe {type(self.pipe)} returns "
                f"the same batch size for all nested examples."
            )

        # concatenate and return
        return {key: reconcat([eg[key] for eg in egs], types[key]) for key in keys}

    @classmethod
    def instantiate_test(cls, **kwargs) -> None:
        return cls(Identity(), **kwargs)


class Nested(ApplyAsFlatten):
    """Apply a pipe to each nested value up to dimension `level`.
    This can be use to modify the nested field inplace  (i.e. sorting, deleting).
    However the all pipe output must have the same batch size.
    """

    def __init__(self, pipe: Pipe | Callable, level=1, **kwargs):
        """
        Args:
            pipe (:obj:`Pipe` or :obj:`Callable`): The pipe to apply to each nested value.
            level (:obj:`int`, `optional`, defaults to 1): The level of nesting to apply
                the pipe to.
        """
        if level == 0:
            super().__init__(pipe=pipe, level=0, **kwargs)
        else:
            pipe = NestedLevel1(pipe)
            super().__init__(pipe=pipe, level=level - 1, **kwargs)

    @classmethod
    def instantiate_test(cls, **kwargs) -> None:
        return cls(Identity(), **kwargs)


class Expand(Pipe):
    """Expand the batch to match the new shape. New dimensions are repeated."""

    def __init__(self, axis: int, *, n: int, **kwargs):
        """
        Args:
            axis (:obj:`int`): The axis to expand.
            n (:obj:`int`): The number of times to repeat the batch.
        """
        super().__init__(**kwargs)
        self.axis = axis
        self.n = n

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        return {
            k: expand_and_repeat(v, axis=self.axis, n=self.n) for k, v in batch.items()
        }

    @classmethod
    def instantiate_test(cls, **kwargs) -> None:
        return cls(1, n=1, **kwargs)
