from __future__ import annotations

from functools import partial
from typing import Optional
import pytest

from pathlib import Path
import tensorstore as ts
import datasets
import numpy as np
import string
import rich
import random
import torch
import tempfile
from torch import nn
from warp_pipes.pipes.collate import Collate
from warp_pipes.support.datastruct import Batch
from warp_pipes.support.shapes import infer_batch_size
from warp_pipes.support.caching import cache_or_load_vectors
from warp_pipes.support.tensorstore_callback import IDX_COL


class DummyModel(nn.Module):
    def __init__(
        self,
        hdim: int = 8,
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.hdim = hdim
        self.linear = nn.Linear(hdim, hdim)
        self.input_key = input_key
        self.output_key = output_key

    def forward(self, batch: Batch) -> torch.Tensor:
        if self.input_key is not None:
            batch = batch[self.input_key]

        output = self.linear(batch)

        if self.output_key is not None:
            output = {self.output_key: output}

        return output


base_cfg = {
    "driver": "zarr",
    "dtype": "float32",
    "input_key": "data",
    "output_key": None,
    "num_workers": 0,
}


def collate_fn(egs, input_key="data", **kwargs):
    inputs = [eg[input_key] for eg in egs]
    inputs = list(map(torch.tensor, inputs))
    return {input_key: torch.stack(inputs)}


@torch.inference_mode()
@pytest.mark.parametrize(
    "cfg",
    [
        {**base_cfg},
        {**base_cfg, "dtype": "float16"},
        {**base_cfg, "driver": "n5"},
        {**base_cfg, "output_key": "vector"},
        {**base_cfg, "num_workers": 2},
    ],
)
def test_cache_or_load_vectors(cfg):
    # prep data and model
    data = np.random.randn(100, 8).astype(np.float32)
    dataset = datasets.Dataset.from_dict({cfg["input_key"]: [d for d in data]})
    model = DummyModel(
        data.shape[1], input_key=cfg["input_key"], output_key=cfg["output_key"]
    )

    # make predictions
    preds = model({cfg["input_key"]: torch.from_numpy(data)})
    if cfg["output_key"] is not None:
        preds = preds[cfg["output_key"]]
    preds = preds.detach().numpy()

    # cache predictions
    with tempfile.TemporaryDirectory() as cache_dir:
        store = cache_or_load_vectors(
            dataset,
            model,
            # trainer: Optional[Trainer] = None,
            model_output_keys=[cfg["output_key"]]
            if cfg["output_key"] is not None
            else None,
            collate_fn=partial(collate_fn, input_key=cfg["input_key"]),
            loader_kwargs={
                "batch_size": 10,
                "num_workers": cfg["num_workers"],
                "pin_memory": False,
            },
            cache_dir=cache_dir,
            target_file=None,
            driver=cfg["driver"],
            dtype=cfg["dtype"],
        )

        # compare the cached predictions with the original predictions
        cached_preds = store.read().result()
        assert cached_preds.dtype == np.dtype(cfg["dtype"])
        assert cached_preds.shape == preds.shape
        assert np.allclose(cached_preds.astype(np.float16), preds.astype(np.float16))
