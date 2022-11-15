from copy import copy
from copy import deepcopy
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from warp_pipes.core.condition import Condition
from warp_pipes.core.condition import Static
from warp_pipes.core.pipe import Pipe
from warp_pipes.support.datastruct import Batch
from warp_pipes.support.datastruct import Eg
from warp_pipes.support.functional import indentity
from warp_pipes.support.json_struct import apply_to_json_struct


class Identity(Pipe):
    """A pipe that passes a batch without modifying it."""

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        return batch

    def _call_egs(self, batch: Batch, **kwargs) -> Batch:
        return batch

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls()


class Lambda(Pipe):
    """Apply a lambda function to the batch."""

    def __init__(
        self,
        op: Callable,
        allow_kwargs: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.op = op
        self.allow_kwargs = allow_kwargs

    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, **kwargs
    ) -> Batch:
        return self._call_all(batch, **kwargs)

    def _call_egs(
        self, examples: List[Eg], idx: Optional[List[int]] = None, **kwargs
    ) -> Batch:
        return self._call_all(examples, **kwargs)

    def _call_all(self, batch: Union[List[Eg], Batch], **kwargs) -> Batch:
        if not self.allow_kwargs:
            kwargs = {}
        return self.op(batch, **kwargs)

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls(op=indentity)


class GetKey(Pipe):
    """Returns a batch containing only the target key."""

    def __init__(self, key: str, **kwargs):
        super().__init__(**kwargs)
        self.key = key

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        return {self.key: batch[self.key]}

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls(key="a")


class FilterKeys(Identity):
    """Filter the keys in the batch given the `Condition` object."""

    _allows_update = False

    def __init__(self, condition: Optional[Condition], **kwargs):
        assert kwargs.get("input_filter", None) is None, "input_filter is not allowed"
        super().__init__(input_filter=condition, **kwargs)

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls(Static(True))


class DropKeys(Pipe):
    """Drop the keys in the current batch."""

    _allows_update = False
    _allows_input_filter = False

    def __init__(
        self,
        keys: Optional[List[str]] = None,
        condition: Optional[Condition] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if keys is None and condition is None:
            raise ValueError("Either keys or condition must be provided")
        if keys is None:
            keys = []
        self.keys = keys
        if condition is None:
            condition = Static(False)
        self.condition = condition

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        keys = list(batch.keys())
        for key in keys:
            if key in self.keys or self.condition(key):
                batch.pop(key)
        return batch

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls(["a"])


class AddPrefix(Pipe):
    """Append the keys with a prefix."""

    _allows_update = False

    def __init__(self, prefix: str, **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        return {f"{self.prefix}{k}": v for k, v in batch.items()}

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls("prefix")


class ReplaceInKeys(Pipe):
    """Remove a pattern `a` with `b` in all keys"""

    _allows_update = False

    def __init__(self, a: str, b: str, **kwargs):
        """
        Args:
            a (:obj:`str`): The pattern to be replaced
            b (:obj:`str`): The pattern to replace with
        """
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        return {k.replace(self.a, self.b): v for k, v in batch.items()}

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls("a", "b")


class RenameKeys(Pipe):
    """Rename a set of keys using a dictionary"""

    def __init__(self, keys: Dict[str, str], **kwargs):
        super().__init__(**kwargs)
        self.keys = keys

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        output = {}
        for old_key, new_key in self.keys.items():
            if old_key in batch:
                output[new_key] = batch[old_key]

        return output

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls({"a": "b"})


class Apply(Pipe):
    """Transform the values in a batch using the transformations registered in `ops`
    registered in `ops`: key, transformation`.
    The argument `element_wise` allows to process each value in the batch element wise.
    """

    _allows_update = False

    def __init__(self, ops: Dict[str, Callable], element_wise: bool = False, **kwargs):
        """
        Args:
            ops (:obj:`Dict[str, Callable]`): A dictionary of key, transformation
            element_wise (:obj:`bool`, `optional`, defaults to :obj:`False`): If True,
                the transformation is applied element wise
        """
        super().__init__(**kwargs)
        self.ops = ops
        self.element_wise = element_wise

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        for key, op in self.ops.items():
            values = batch[key]
            if self.element_wise:
                batch[key] = apply_to_json_struct(values, op)
            else:
                batch[key] = op(values)

        return batch

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls({"a": indentity})


class ApplyToAll(Pipe):
    """Apply a transformation
    registered in `ops`: key, transformation`.
    The argument `element_wise` allows to process each value in the batch element wise.
    """

    _allows_update = False

    def __init__(
        self,
        op: Callable,
        element_wise: bool = False,
        allow_kwargs: bool = False,
        **kwargs,
    ):
        """
        Args:
            op (:obj:`Callable`): A transformation to be applied to each batch attribute
            element_wise (:obj:`bool`, `optional`, defaults to :obj:`False`): If True,
                the transformation is applied element wise
            allow_kwargs (:obj:`bool`, `optional`, defaults to :obj:`False`): If True,
                the transformation is allowed to receive kwargs
        """
        super().__init__(**kwargs)
        self.op = op
        self.element_wise = element_wise
        self.allow_kwargs = allow_kwargs

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        if not self.allow_kwargs:
            kwargs = {}
        for key, values in batch.items():
            if self.element_wise:
                batch[key] = [self.op(x, **kwargs) for x in values]
            else:
                batch[key] = self.op(values, **kwargs)
        return batch

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls(indentity)


class CopyBatch(Pipe):
    """Copy an input batch"""

    _allows_update = False
    _allows_input_filter = False

    def __init__(self, *, deep: bool = False, **kwargs):
        """
        Args:
            deep (:obj:`bool`, `optional`, defaults to :obj:`False`): If True, copy the
                batch using deepcopy
        """
        super().__init__(**kwargs)
        self.deep = deep

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        if self.deep:
            return deepcopy(batch)
        else:
            return copy(batch)

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls()


class Partial(Pipe):
    """Run a pipe with keyword arguments"""

    def __init__(
        self,
        pipe: Pipe,
        *,
        pipe_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Args:
            pipe (:obj:`Pipe`): The pipe to be run
            pipe_kwargs (:obj:`Dict[str, Any]`, `optional`): The keyword arguments to be
                passed to the pipe
        """
        super().__init__(**kwargs)
        self.pipe = pipe
        if pipe_kwargs is None:
            pipe_kwargs = {}
        self.pipe_kwargs = pipe_kwargs

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        _kwargs = self.pipe_kwargs.copy()
        _kwargs.update(kwargs)
        return self.pipe(batch, **_kwargs)

    def _call_egs(
        self, examples: List[Eg], idx: Optional[List[int]] = None, **kwargs
    ) -> Batch:
        _kwargs = self.kwargs.copy()
        _kwargs.update(kwargs)
        return self.pipe(examples, **_kwargs)

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls(indentity, pipe_kwargs={"a": "a"})
