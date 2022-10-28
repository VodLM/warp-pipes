from __future__ import annotations

from typing import Optional, List
import torch
import numpy as np

from warp_pipes.support.datastruct import Batch

class DummyModel(torch.nn.Module):
    def __init__(
        self,
        hdim: int = 8,
        input_keys: Optional[str | List] = None,
        output_key: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.hdim = hdim
        self.linear = torch.nn.Linear(hdim, hdim)
        if isinstance(input_keys, str):
            input_keys = [input_keys]
        self.input_keys = input_keys
        self.output_key = output_key

    def forward(self, batch: Batch, **kwargs) -> torch.Tensor:
        if self.input_keys is not None:
            match_keys = [k for k in batch.keys() if k in self.input_keys]
            if len(match_keys) != 1:
                raise ValueError(f"Found {len(match_keys)} != 1 matched "
                                 f"input keys (matches: {match_keys}, "
                                 f"input keys: {batch.keys()}).")
            batch = batch[match_keys[0]]

        output = self.linear(batch)

        if self.output_key is not None:
            output = {self.output_key: output}

        return output
