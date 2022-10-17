from typing import Iterable
from typing import List
from typing import Optional

from warp_pipes.core.condition import In
from warp_pipes.core.pipe import Pipe
from warp_pipes.pipes.basics import Identity
from warp_pipes.support.datastruct import Batch
from warp_pipes.support.datastruct import Eg


class Collate(Pipe):
    """
    Create a Batch object from a list of examples, where an
    example is defined as a batch of one element.

    This default class concatenate values as lists.
    """

    _allows_update = False

    def __init__(self, keys: Optional[List[str]] = None, **kwargs):
        if keys is not None:
            msg = "input_filter is not allowed when keys are explicitely set"
            assert kwargs.get("input_filter", None) is None, msg
            input_filter = In(keys)
        else:
            input_filter = kwargs.pop("input_filter", None)
        super().__init__(**kwargs, input_filter=input_filter)

    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, **kwargs
    ) -> Batch:
        return batch

    def _call_egs(
        self, examples: List[Eg], idx: Optional[List[int]] = None, **kwargs
    ) -> Batch:
        first_eg = examples[0]
        keys = set(first_eg.keys())
        return {key: [eg[key] for eg in examples] for key in keys}

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls(**kwargs)


class DeCollate(Pipe):
    """Returns a list of examples from a batch"""

    _allows_update = False

    def _call_batch(self, batch: Batch, **kwargs) -> List[Eg]:
        keys = list(batch.keys())
        length = len(batch[keys[0]])
        lengths = {k: len(v) for k, v in batch.items()}
        assert all(
            length == eg_l for eg_l in lengths.values()
        ), f"un-equal lengths: {lengths}"
        return [{k: batch[k][i] for k in keys} for i in range(length)]

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls(**kwargs)


class FirstEg(Pipe):
    """Returns the first example"""

    _allows_update = False

    def _call_egs(self, examples: List[Eg], **kwargs) -> Eg:
        return examples[0]

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls(**kwargs)


class ApplyToEachExample(Pipe):
    _allows_update = False

    def __init__(self, pipe: Pipe, **kwargs):
        super(ApplyToEachExample, self).__init__(**kwargs)
        self.pipe = pipe

    def _call_egs(self, examples: List[Eg], **kwargs) -> Iterable[Eg]:
        for eg in examples:
            yield self.pipe(eg, **kwargs)

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls(Identity(), **kwargs)
