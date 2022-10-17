from __future__ import annotations

import shutil
from numbers import Number
from typing import List
from typing import Optional

import numpy as np
import rich
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from warp_pipes.core.pipe import Pipe
from warp_pipes.support.datastruct import Batch
from warp_pipes.support.datastruct import Eg
from warp_pipes.support.nesting import flatten_nested
from warp_pipes.support.pretty import get_separator
from warp_pipes.support.pretty import pprint_batch
from warp_pipes.support.shapes import infer_shape


class PrintBatch(Pipe):
    """Print a batch of data. This is useful for debugging purposes."""

    def __init__(
        self, header: Optional[str] = None, report_nans: bool = False, **kwargs
    ):
        super(PrintBatch, self).__init__(**kwargs)
        self.header = header
        self.report_nans = report_nans

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        header = self.header
        if header is None:
            header = "PrintBatch"
        if self.id is not None:
            header = f"{header} (id={self.id})"
        pprint_batch(batch, header=header, report_nans=self.report_nans)
        if len(kwargs):
            kwargs = {
                k: self._format_kwarg_v(v) for k, v in kwargs.items() if v is not None
            }
            rich.print(f"PrintBatch input kwargs = {kwargs}")
        return batch

    @staticmethod
    def _format_kwarg_v(v):
        u = str(type(v))
        if isinstance(v, list):
            u += f" (length={len(v)})"
        elif isinstance(v, (np.ndarray, Tensor)):
            u += f" (shape={v.shape})"
        elif isinstance(v, Number):
            u += f" (value={v})"

        return u

    def _call_egs(self, examples: List[Eg], **kwargs) -> List[Eg]:
        header = f"{self.header} : " if self.header is not None else ""
        try:
            pprint_batch(examples[0], header=f"{header}First example")
        except Exception:
            rich.print(f"#{header}Failed to print using pprint_batch. First Example:")
            rich.print(examples[0])

        return examples

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls(**kwargs)


class PrintContent(Pipe):
    """Print the raw content of the batch"""

    def __init__(
        self,
        keys: str | List[str],
        n: Optional[int] = None,
        decode_keys: List[str] | bool = None,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        header: Optional[str] = None,
        **kwargs,
    ):
        super(PrintContent, self).__init__(**kwargs)
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        if isinstance(decode_keys, bool):
            decode_keys = keys if decode_keys else []
        if decode_keys is None:
            decode_keys = []
        self.decode_keys = decode_keys
        self.n = n
        self.header = header
        self.tokenizer = tokenizer

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        console_width, _ = shutil.get_terminal_size()
        header = self.header
        if header is None:
            header = "Batch Content"
        header = f"  {header}  "
        rich.print(f"{header:=^{console_width}}")
        for key in self.keys:
            feature = batch.get(key, None)
            shape = infer_shape(feature)
            feature_info = f"  {key}:{type(feature)} ({shape})  "
            rich.print(f"{feature_info:-^{console_width}}")
            feature = flatten_nested(feature, level=max(0, len(shape) - 1))
            if self.n:
                feature = feature[: self.n]
            for x in feature:
                if key in self.decode_keys:
                    if self.tokenizer is None:
                        raise ValueError("tokenizer is required to decode")
                    x = self.tokenizer.decode(x, skip_special_tokens=False)
                print(x)
                rich.print(get_separator("."))
        return batch

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls(["a"], **kwargs)
