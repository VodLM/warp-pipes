from typing import Optional
import torch
import numpy as np

from warp_pipes.support.datastruct import Batch

class DummyModel(torch.nn.Module):
    def __init__(
        self,
        hdim: int = 8,
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.hdim = hdim
        self.linear = torch.nn.Linear(hdim, hdim)
        self.input_key = input_key
        self.output_key = output_key

    def forward(self, batch: Batch, **kwargs) -> torch.Tensor:
        if self.input_key is not None:
            batch = batch[self.input_key]

        output = self.linear(batch)

        if self.output_key is not None:
            output = {self.output_key: output}

        return output
