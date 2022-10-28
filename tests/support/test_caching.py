from __future__ import annotations

import tempfile
from functools import partial

import datasets
import numpy as np
import pytest
import torch
from pytorch_lightning import Trainer

from tests.utils.dummy_model import DummyModel
from warp_pipes.support.caching import cache_or_load_vectors, CacheConfig

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
        data.shape[1], input_keys=cfg["input_key"], output_key=cfg["output_key"]
    )

    # make predictions
    preds = model({cfg["input_key"]: torch.from_numpy(data)})
    if cfg["output_key"] is not None:
        preds = preds[cfg["output_key"]]
    preds = preds.detach().numpy()

    # cache predictions
    with tempfile.TemporaryDirectory() as cache_dir:
        cache_config = CacheConfig(
                model_output_key=cfg["output_key"]
                if cfg["output_key"] is not None
                else None,
                collate_fn=partial(collate_fn, input_key=cfg["input_key"]),
                loader_kwargs={
                    "batch_size": 10,
                    "num_workers": cfg["num_workers"],
                    "pin_memory": False,
                },
                driver=cfg["driver"],
                dtype=cfg["dtype"],
            )

        store = cache_or_load_vectors(
            dataset,
            model,
            cache_dir=cache_dir,
            # trainer: Optional[Trainer] = None,
            config=cache_config
        )

        # compare the cached predictions with the original predictions
        cached_preds = store.read().result()
        assert cached_preds.dtype == np.dtype(cfg["dtype"])
        assert cached_preds.shape == preds.shape
        assert np.allclose(cached_preds.astype(np.float16), preds.astype(np.float16))
