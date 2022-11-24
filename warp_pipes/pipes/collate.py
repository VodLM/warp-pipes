from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from warp_pipes.core.condition import Contains
from warp_pipes.core.condition import HasKeys
from warp_pipes.core.condition import HasKeyWithPrefix
from warp_pipes.core.condition import HasPrefix
from warp_pipes.core.condition import In
from warp_pipes.core.condition import Not
from warp_pipes.core.condition import Reduce
from warp_pipes.core.pipe import Pipe
from warp_pipes.pipes.basics import AddPrefix
from warp_pipes.pipes.basics import ApplyToAll
from warp_pipes.pipes.basics import Identity
from warp_pipes.pipes.basics import ReplaceInKeys
from warp_pipes.pipes.nesting import ApplyAsFlatten
from warp_pipes.pipes.pipelines import Gate
from warp_pipes.pipes.pipelines import Parallel
from warp_pipes.pipes.pipelines import Sequential
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


def to_tensor_op(inputs: List[Any]) -> Tensor:
    if isinstance(inputs, Tensor):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return torch.from_numpy(inputs)
    else:
        try:
            return torch.tensor(inputs)
        except Exception as exc:
            if isinstance(inputs, list):
                inputs = [to_tensor_op(i) for i in inputs]
                return torch.stack(inputs, dim=0)
            else:
                import rich

                rich.print(inputs)
                raise exc


class Padding(Pipe):
    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        special_padding_tokens: Dict = None,
        **kwargs,
    ):
        super(Padding, self).__init__(**kwargs)
        if special_padding_tokens is None:
            special_padding_tokens = {}
        self.special_padding_tokens = special_padding_tokens
        self.tokenizer = tokenizer

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:

        special_values = {
            k: batch[k] for k in self.special_padding_tokens.keys() if k in batch.keys()
        }
        pad_input = {k: v for k, v in batch.items() if k not in special_values}
        # pad `normal` values using `tokenizer.pad`
        output = self.tokenizer.pad(pad_input, return_tensors="pt")

        # pad the special cases
        for k, v in special_values.items():
            lenght = output["input_ids"].shape[-1]
            output[k] = self._pad(v, lenght, self.special_padding_tokens[k])

        return output

    def _pad(self, x, length, fill_value):
        y = []
        for z in x:
            if len(z) < length:
                z = z + (length - len(z)) * [fill_value]
            y += [torch.tensor(z)]

        return torch.stack(y)

    @classmethod
    def instantiate_test(cls, **kwargs):
        return None


class CollateField(Gate):
    """
    Collate examples for a given field.
    Field corresponds to the prefix of the keys (field.attribute)
    This Pipe is a Gate and is only activated if keys for the field are present.

    This class handles nested examples, which nesting level must be indicated using `level`.
    """

    def __init__(
        self,
        field: str,
        *,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        exclude: Optional[List[str]] = None,
        include_only: Optional[List[str]] = None,
        to_tensor: Optional[List[str]] = None,
        nesting_level: int | List[str] = 0,
        **kwargs,
    ):
        # set defaults
        prefix = f"{field}."
        if exclude is None:
            exclude = []
        if include_only is None:
            include_only = []
        if to_tensor is None:
            to_tensor = []

        # define the input filter
        if len(include_only):
            include_keys = [f"{prefix}{i}" for i in include_only]
            include_only_cond = In(include_keys)
        else:
            include_only_cond = True

        input_filter = Reduce(
            HasPrefix(prefix),
            include_only_cond,
            *[Not(Contains(f"{prefix}{e}")) for e in exclude],
            reduce_op=all,
        )

        # pipe used to tensorize values
        if len(to_tensor):
            tensorizer_pipe = ApplyToAll(
                op=to_tensor_op, allow_kwargs=False, input_filter=In(to_tensor)
            )
            tensorizer_pipe = ApplyAsFlatten(tensorizer_pipe, level=nesting_level)
        else:
            tensorizer_pipe = None

        # pipe used to pad and collate tokens
        if tokenizer is not None:
            tokenizer_pipe = Gate(
                HasKeys(["input_ids"]),
                pipe=ApplyAsFlatten(Padding(tokenizer=tokenizer), level=nesting_level),
                input_filter=In(
                    ["input_ids", "attention_mask", "offset_mapping", "token_type_ids"]
                ),
                id="pad-and-collate-tokens",
            )
        else:
            tokenizer_pipe = None

        # define the body of the pipe, all pipe bellow operates without the prefix.
        if tokenizer_pipe is not None or tensorizer_pipe is not None:
            body = Sequential(
                ReplaceInKeys(prefix, ""),
                Parallel(tensorizer_pipe, tokenizer_pipe, update=True),
                AddPrefix(prefix),
            )
        else:
            body = None

        super(CollateField, self).__init__(
            condition=HasKeyWithPrefix(prefix),
            pipe=Sequential(
                Collate(),
                body,
                input_filter=input_filter,
            ),
            **kwargs,
        )

    @classmethod
    def instantiate_test(cls, **kwargs):
        return cls(field=str(), **kwargs)
