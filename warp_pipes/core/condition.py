import abc
from typing import Any
from typing import Callable
from typing import List
from typing import Union

try:
    from functools import singledispatchmethod
except Exception:
    from singledispatchmethod import singledispatchmethod

from warp_pipes.core.fingerprintable import Fingerprintable
from warp_pipes.support.datastruct import Batch


class Condition(Fingerprintable):
    """This class implements a condition used for control flow within Pipes."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, x: Any, **kwargs) -> bool:
        """Returns True if the input matches the condition."""
        raise NotImplementedError


class Contains(Condition):
    """check if the key is in the set of `allowed_keys`"""

    def __init__(self, pattern: str, **kwargs):
        super().__init__(**kwargs)
        self.pattern = pattern

    def __call__(self, x: Any, **kwargs) -> bool:
        return self.pattern in x


class In(Condition):
    """check if the key is in the set of `allowed_keys`"""

    def __init__(self, allowed_values: List[str], **kwargs):
        super(In, self).__init__(**kwargs)
        self.allowed_keys = allowed_values

    def __call__(self, x: Any, **kwargs) -> bool:
        return x in self.allowed_keys

    def __repr__(self):
        return f"{self.__class__.__name__}({self.allowed_keys})"


class HasPrefix(Condition):
    """check whether the value starts with a given prefix"""

    def __init__(self, prefix: str, **kwargs):
        super(HasPrefix, self).__init__(**kwargs)
        self.prefix = prefix

    def __call__(self, x: Any, **kwargs) -> bool:
        return str(x).startswith(self.prefix)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.prefix})"


class Reduce(Condition):
    """Reduce multiple conditions into one outcome."""

    def __init__(
        self, *conditions: Union[bool, Condition], reduce_op: Callable = all, **kwargs
    ):
        super(Reduce, self).__init__(**kwargs)
        self.reduce_op = reduce_op
        self.conditions = list(conditions)

    def __call__(self, x: Any, **kwargs) -> bool:
        def safe_call_c(c: Union[bool, Condition], x: Any) -> bool:
            if isinstance(c, Condition):
                return c(x)
            elif isinstance(c, bool):
                return c
            else:
                raise TypeError(f"{c} is not a valid condition")

        return self.reduce_op(safe_call_c(c, x) for c in self.conditions)

    def __repr__(self):
        return f"{self.__class__.__name__}(conditions={list(self.conditions)}, op={self.reduce_op})"


class Not(Condition):
    """`not` operator."""

    def __init__(self, condition: Condition, **kwargs):
        super(Not, self).__init__(**kwargs)
        self.condition = condition

    def __call__(self, x: Any, **kwargs) -> bool:
        return not self.condition(x)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.condition})"


class Static(Condition):
    """Static boolean outcome."""

    def __init__(self, cond: bool, **kwargs):
        super(Static, self).__init__(**kwargs)
        self.cond = cond

    def __call__(self, x: Any, **kwargs) -> bool:
        return self.cond

    def __repr__(self):
        return f"{self.__class__.__name__}({self.cond})"


class BatchCondition(Condition):
    """Condition operating on the batch level."""

    __metaclass__ = abc.ABCMeta

    @singledispatchmethod
    def __call__(self, batch: Batch) -> bool:
        raise TypeError(f"Cannot handle input of type type {type(batch)}.")

    @__call__.register(dict)
    def _(self, batch: Batch) -> bool:
        return self._call_batch(batch)

    @__call__.register(list)
    def _(self, batch: List[Batch]) -> bool:
        return self._call_egs(batch)

    @abc.abstractmethod
    def _call_batch(self, batch: Batch, **kwargs) -> bool:
        raise NotImplementedError

    def _call_egs(self, batch: list) -> bool:
        first_eg = batch[0]
        return self._call_batch(first_eg)


class HasKeyWithPrefix(BatchCondition):
    """Test if the batch contains at least one key with the specified prefix"""

    def __init__(self, prefix: str, **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix

    def _call_batch(self, batch: Batch, **kwargs) -> bool:
        return any(str(k).startswith(self.prefix) for k in batch.keys())

    def __repr__(self):
        return f"{self.__class__.__name__}({self.prefix})"


class HasKeys(BatchCondition):
    """Test if the batch contains all the required keys"""

    def __init__(self, keys: List[str], **kwargs):
        super().__init__(**kwargs)
        self.keys = keys

    def _call_batch(self, batch: Batch, **kwargs) -> bool:
        return all(key in batch for key in self.keys)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.keys})"


class AllValuesOfType(BatchCondition):
    """Check if all batch values are of the specified type"""

    def __init__(self, cls: type, **kwargs):
        super().__init__(**kwargs)
        self.cls = cls

    def _call_batch(self, batch: Batch, **kwargs) -> bool:
        return all(isinstance(v, self.cls) for v in batch.values())

    def __repr__(self):
        return f"{self.__class__.__name__}({self.cls.__name__})"
