from __future__ import annotations

import tempfile
from copy import copy
from os import PathLike
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import rich

try:
    from functools import singledispatchmethod
except Exception:
    from singledispatchmethod import singledispatchmethod

import pytorch_lightning as pl
import tensorstore as ts
import torch
from datasets import Dataset
from datasets import DatasetDict
from pytorch_lightning.utilities import move_data_to_device
from torch import nn


from warp_pipes.core.pipe import Pipe
from warp_pipes.pipes.basics import Identity
from warp_pipes.support.datastruct import Batch

from warp_pipes.support.fingerprint import get_fingerprint
from warp_pipes.support import caching

from loguru import logger


class PredictWithoutCache(Pipe):
    """Process batches of data through a model and return the output."""

    def __init__(
        self,
        model: pl.LightningModule | nn.Module | Callable,
        **kwargs,
    ):
        super(PredictWithoutCache, self).__init__(**kwargs)
        self.model = model

    @torch.inference_mode()
    def _call_batch(
        self,
        batch: Batch,
        **kwargs,
    ) -> Batch:
        """Process a batch of data through the model and return the output."""
        if isinstance(self.model, nn.Module):
            device = next(iter(self.model.parameters())).device
            batch = move_data_to_device(batch, device)

        model_output = self.model(batch, **kwargs)
        return model_output

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls(Identity(), **kwargs)


class PredictWithCache(Pipe):
    """ "Processand cache a dataset with a model, retrieve the cached predictions
    when calling the Pipe."""

    no_fingerprint: Optional[List[str]] = ["stores", "caching_kwargs"]

    def __init__(
        self,
        model: pl.LightningModule | nn.Module | Callable,
        caching_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        super(PredictWithCache, self).__init__(**kwargs)
        self.model = model
        self.stores: Dict[str, ts.TensorStore] = {}

        # get the argument for caching the dataset
        if "model_output_key" not in caching_kwargs:
            raise ValueError(
                "`model_output_key` must be provided in `caching_kwargs` to cache the predictions."
            )
        self.caching_kwargs = caching_kwargs
        self.model_output_key = caching_kwargs["model_output_key"]

    def _call_batch(
        self,
        batch: Batch,
        idx: Optional[List[int]] = None,
        cache_fingerprint: Optional[str] = None,
        **kwargs,
    ) -> Batch:

        if idx is None:
            raise ValueError("idx must be provided to retrieve the cached predictions.")

        if cache_fingerprint is None:
            raise ValueError(
                "cache_fingerprint must be provided to "
                "retrieve the cached predictions. "
                "Use `pipe.get_cache_fingerprint(dataset)` to get the fingerprint "
                "of the dataset."
            )

        if cache_fingerprint not in self.stores:
            raise ValueError(
                f"cache_fingerprint {cache_fingerprint} is not in the cache. "
                f"Known fingerprints: {list(self.stores.keys())}"
            )

        store = self.stores[cache_fingerprint]
        assert isinstance(store, ts.TensorStore)
        # if isinstance(store, (str, Path)):
        #     # TODO: should not happen. Remove this when the bug is fixed.
        #     store = caching.load_store(store)
        #     self.stores[cache_fingerprint] = store
        cached_predictions = store[idx].read().result()

        return {self.model_output_key: cached_predictions}

    def _call_dataset(
        self,
        dataset: Dataset,
        *,
        fingerprint_kwargs_exclude: Optional[List[str]] = None,
        **kwargs,
    ) -> Dataset:
        if fingerprint_kwargs_exclude is None:
            fingerprint_kwargs_exclude = []
        fingerprint_kwargs_exclude.append("caching_kwargs")

        if "cache_fingerprint" in kwargs:
            logger.warning("Overriding kwargs cache_fingerprint.")
        kwargs["cache_fingerprint"] = self.get_cache_fingerprint(dataset)
        return super()._call_dataset(
            dataset,
            fingerprint_kwargs_exclude=fingerprint_kwargs_exclude,
            **kwargs,
        )

    @singledispatchmethod
    def cache(
        self,
        dataset: Dataset,
        *,
        caching_kwargs: Dict[str, Any] = None,
    ) -> str:
        """
        Cache the predictions of the model on the dataset.

        Args:
            dataset (:obj:`Dataset`): The dataset to cache.
            caching_kwargs (:obj:`Dict[str, Any]`, `optional`): The arguments to pass to
                the caching function.

        Returns:
            :obj:`str`: fingerprint of the cache.

        """
        # define the arguments for caching the dataset
        if caching_kwargs is None:
            caching_kwargs = {}
        if "model_output_key" in caching_kwargs:
            if caching_kwargs["model_output_key"] != self.model_output_key:
                raise ValueError(
                    f"Cannot change the `model_output_key`. Received: "
                    f"{caching_kwargs['model_output_key']}, "
                    f"registered: {self.model_output_key}"
                )
        _caching_kwargs = copy(self.caching_kwargs)
        _caching_kwargs.update(caching_kwargs)

        # define a unique fingerprint for the cache file
        cache_fingerprint = self.get_cache_fingerprint(dataset)

        # cache the dataset
        store = caching.cache_or_load_vectors(dataset, self.model, **_caching_kwargs)
        # store the tensorstore
        self.stores[cache_fingerprint] = store

        return cache_fingerprint

    @cache.register(DatasetDict)
    def _(self, dataset: DatasetDict, **kwargs) -> Dict[str, ts.TensorStore]:
        """Cache the predictions of the model for a DatasetDict."""
        return {split: self.cache(dset, **kwargs) for split, dset in dataset.items()}

    def get_cache_fingerprint(self, dataset: Dataset) -> str:
        """Fingerprint the model and the dataset to get a unique identifier."""
        fingerprint = get_fingerprint(
            {
                "model": get_fingerprint(self.model),
                "model_output_key": self.model_output_key,
                "dataset": dataset._fingerprint,
            }
        )
        return fingerprint

    @staticmethod
    def _setup_caching_kwargs(caching_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Adapt the default cacheing kwargs"""
        if caching_kwargs is None:
            caching_kwargs = {}
        if "cache_dir" not in caching_kwargs and "target_path" not in caching_kwargs:
            logger.warning(
                "No cache_dir or target_path specified, using a temporary directory."
            )
            caching_kwargs["cache_dir"] = tempfile.mkdtemp()
        return caching_kwargs

    def __getstate__(self):
        state = copy(super().__getstate__())

        for key, store in state["stores"].items():
            if isinstance(store, ts.TensorStore):
                state["stores"][key] = str(
                    caching.infer_path_from_store(store).absolute()
                )

        rich.print(f"> getstate: {state}")

        return state

    def __setstate__(self, state):
        state = copy(state)

        rich.print(f"> setstate: {state}")

        for key, store_path in state["stores"].items():
            state["stores"][key] = caching.load_store(store_path, read=True, open=True)

        rich.print("> setstate: DONE")
        rich.print(f">>>> setstate: {state['stores']}")

        super().__setstate__(state)
