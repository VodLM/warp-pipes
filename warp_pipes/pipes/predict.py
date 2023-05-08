from __future__ import annotations

import os
from copy import copy
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

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
        for key, value in model_output.items():
            if isinstance(value, torch.Tensor):
                model_output[key] = value.cpu().numpy()
        return model_output

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls(Identity(), **kwargs)


class PredictWithCache(Pipe):
    """ "Process and cache a dataset with a model, retrieve the cached predictions
    when calling the Pipe."""

    _no_fingerprint: Optional[List[str]] = ["stores", "cache_config"]

    def __init__(
        self,
        trainer: pl.Trainer,
        model: pl.LightningModule | nn.Module | Callable,
        *,
        cache_dir: Path,
        cache_config: Dict | caching.CacheConfig,
        **kwargs,
    ):
        super(PredictWithCache, self).__init__(**kwargs)
        self.model = model
        self.cache_dir = cache_dir
        self.stores: Dict[str, ts.TensorStore] = {}
        if not isinstance(cache_config, caching.CacheConfig):
            cache_config = caching.CacheConfig(**cache_config)

        # get the argument for caching the dataset
        try:
            model_output_key = cache_config.model_output_key
        except KeyError:
            raise ValueError(
                "`model_output_key` must be provided in `cache_config` "
                "to cache the predictions."
            )
        self.cache_config = cache_config
        self.model_output_key = model_output_key

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
        if not isinstance(store, ts.TensorStore):
            raise TypeError(f"store must be a TensorStore, got {type(store)}")
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
        cache_config: Dict | caching.CacheConfig = None,
        return_store: bool = False,
    ) -> ts.TensorStore | str:
        """
        Cache the predictions of the model on the dataset.

        Args:
            dataset (:obj:`Dataset`): The dataset to cache.
            cache_config (:obj:`caching.CacheConfig`, `optional`): The arguments to configuration to
            pass to the caching function.

        Returns:
            :obj:`str`: fingerprint of the cache.

        """
        # define a unique fingerprint for the cache file
        cache_fingerprint = self.get_cache_fingerprint(dataset)

        # update self.cache_config with the given cache_config
        cache_config_ = self.cache_config.copy(update=cache_config)
        assert cache_config_.model_output_key == self.model_output_key

        # cache the dataset
        store = caching.cache_or_load_vectors(
            dataset,
            self.trainer,
            self.model,
            cache_dir=self.cache_dir,
            config=cache_config_,
        )
        # store the tensorstore
        self.stores[cache_fingerprint] = store

        if return_store:
            return store
        else:
            return cache_fingerprint

    def get_cached_vectors(self, dataset: Dataset, **kwargs) -> ts.TensorStore:
        """Get the vectors from a dataset."""
        cache_fingerprint = self.get_cache_fingerprint(dataset)
        if cache_fingerprint not in self.stores:
            raise ValueError(
                f"cache_fingerprint {cache_fingerprint} is not in the cache. "
                f"Known fingerprints: {list(self.stores.keys())}"
            )

        store = self.stores[cache_fingerprint]
        if isinstance(store, os.PathLike):
            store = caching.load_store(path=store, read=True)
        return store

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

    def __getstate__(self):
        state = copy(super().__getstate__())

        new_state_stores = {}
        for key, store in state["stores"].items():
            if isinstance(store, ts.TensorStore):
                new_state_stores[key] = str(
                    caching.infer_path_from_store(store).absolute()
                )
        state["stores"] = new_state_stores

        return state

    def __setstate__(self, state):
        state = copy(state)

        new_state_stores = {}
        for key, store_path in state["stores"].items():
            store = caching.load_store(store_path)
            new_state_stores[key] = store
        state["stores"] = new_state_stores

        super().__setstate__(state)

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls(
            Identity(),
            cache_config={
                "model_output_key": str(),
            },
            **kwargs,
        )


class Predict(PredictWithCache):
    def __init__(
        self,
        trainer: pl.Trainer,
        model: pl.LightningModule | nn.Module | Callable,
        requires_cache: bool = False,
        **kwargs,
    ):
        super(Predict, self).__init__(model=model, trainer=trainer, **kwargs)
        self.requires_cache = requires_cache

    def _call_batch(
        self,
        batch: Batch,
        idx: Optional[List[int]] = None,
        cache_fingerprint: Optional[str] = None,
        **kwargs,
    ) -> Batch:

        use_cache = self.requires_cache or cache_fingerprint in self.stores
        if use_cache:
            return PredictWithCache._call_batch(
                self,
                batch,
                idx=idx,
                cache_fingerprint=cache_fingerprint,
                **kwargs,
            )
        else:
            return PredictWithoutCache._call_batch(
                self,
                batch,
                idx=idx,
                cache_fingerprint=cache_fingerprint,
                **kwargs,
            )
