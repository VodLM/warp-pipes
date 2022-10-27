from __future__ import annotations

import os
import tempfile
from copy import copy
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import TypeVar

import pytorch_lightning as pl
from datasets import Dataset
from datasets import DatasetDict
from loguru import logger
from omegaconf import DictConfig
from torch import nn

from warp_pipes.pipes import Pipe
from warp_pipes.pipes import Predict
from warp_pipes.support import caching
from warp_pipes.support.datastruct import Batch
from warp_pipes.support.search_engines import AutoSearchEngine
from warp_pipes.support.search_engines import SearchEngine
from warp_pipes.support.shapes import infer_batch_shape

HfDataset = TypeVar("HfDataset", Dataset, DatasetDict)

MAX_INDEX_CACHE_AGE = 60 * 60 * 24 * 3  # 3 days
TEMPDIR_SUFFIX = "-tempdir"


def _get_unique(x: List):
    y = list(set(x))
    assert len(y) == 1
    return y[0]


class Index(Pipe):
    """Keep an index of a Dataset and search it using queries."""

    index_name: Optional[str] = None
    is_indexed: bool = False
    default_key: Optional[str | List[str]] = None
    _no_fingerprint: List[str] = [
        "cache_dir",
        "trainer",
        "loader_kwargs",
    ]

    def __init__(
        self,
        corpus: Dataset,
        *,
        cache_dir: os.PathLike = None,
        engines: List[SearchEngine | Dict] = None,
        model: pl.LightningModule | nn.Module = None,
        index_cache_config: Dict | caching.CacheConfig,
        query_cache_config: Dict | caching.CacheConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # setting up cache dir
        if cache_dir is None:
            cache_dir = tempfile.mkdtemp(suffix=TEMPDIR_SUFFIX)
            logger.warning(f"No cache_dir is provided, using {cache_dir}")
        cache_dir = Path(cache_dir) / f"fz-index-{corpus._fingerprint}"

        # register the Engines
        if isinstance(engines, (dict, DictConfig)):
            assert all(isinstance(cfg, (dict, DictConfig)) for cfg in engines.values())
            engines = [{**{"name": k}, **v} for k, v in engines.items()]

        self.engines_config = copy(engines)
        self.engines = []
        for engine in engines:
            engine_path = cache_dir / engine["name"]
            if isinstance(engine, dict):
                engine = AutoSearchEngine(
                    name=engine["name"], path=engine_path, config=engine["config"]
                )
            elif isinstance(engine, SearchEngine):
                if engine.path != engine_path:
                    logger.warning(
                        f"Overriding engine path {engine.path} "
                        f"({type(engine).__name__}) with {engine_path}"
                    )
                    engine.path = engine_path
            else:
                raise TypeError(f"Unknown engine type {type(engine)}")

            if engine.require_vectors and model is None:
                logger.warning(
                    f"No `model` was provided => skipping {type(engine).__name__}"
                )
                continue
            else:
                self.engines.append(engine)

        if len(self.engines) == 0:
            raise ValueError(f"No engines were registered for {type(self).__name__}")

        # Register the model and the pipes used
        # to handle the processing of the data
        self.predict_index = Predict(model, cache_config=index_cache_config)
        self.predict_queries = Predict(model, cache_config=query_cache_config)

        # build the engines and save them to disk
        self.build_engines(corpus)

    def build_engines(self, corpus: Dataset):
        if self.requires_vector:
            vectors = self.predict_index.cache(corpus, return_store=True)
        else:
            vectors = None

        for engine in self.engines:
            engine.build(corpus=corpus, vectors=vectors)
            engine.free_memory()

    @property
    def requires_vector(self):
        return any(engine.require_vectors for engine in self.engines)

    def load(self):
        """Load the index from disk."""
        for engine in self.engines:
            engine.load()

    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, **kwargs
    ) -> Batch:

        if self.requires_vector:
            vectors = self.predict_queries(batch)[self.predict_queries.model_output_key]
        else:
            vectors = None

        for engine in self.engines:
            batch = engine(batch, vectors=vectors, **kwargs)

        return batch

    def _call_dataset(self, dataset: Dataset, **kwargs) -> Dataset:
        return self._call_dataset_any(dataset, **kwargs)

    def _call_dataset_dict(self, dataset: DatasetDict, **kwargs) -> DatasetDict:
        return self._call_dataset_any(dataset, **kwargs)

    def _call_dataset_any(
        self,
        dataset: HfDataset,
        cache_config: Dict = None,
        **kwargs,
    ) -> HfDataset:
        # infer the nesting level. I.e. Questions as shape [bs, ...] or [bs, n_options, ...]
        # cache the query vectors
        if self.requires_vector:
            vectors = self.predict_queries.cache(
                dataset, cache_config=cache_config, return_store=True
            )
            # put the model back to cpu to save memory
            self.predict_queries.model.cpu()
        else:
            vectors = None if isinstance(dataset, Dataset) else {}

        # process the dataset with the Engines
        for engine in self.engines:
            if isinstance(dataset, DatasetDict):
                dataset = DatasetDict(
                    {
                        split: engine(
                            dset,
                            vectors=vectors.get(split, None)
                            if engine.require_vectors
                            else None,
                            desc=f"{type(engine).__name__} ({split})",
                            **kwargs,
                        )
                        for split, dset in dataset.items()
                    }
                )
            elif isinstance(dataset, Dataset):
                dataset = engine(
                    dataset,
                    vectors=vectors if engine.require_vectors else None,
                    # fingerprint_state=fingerprint_state,
                    desc=f"{type(engine).__name__}",
                    **kwargs,
                )
            else:
                raise TypeError(f"Unsupported dataset type: {type(dataset)}")

            # free the CPU and GPU memory allocated by the engine
            engine.free_memory()

        return dataset

    def __repr__(self):
        return f"{type(self).__name__}(engines={self.engines_config})"


def infer_nesting_level(dataset: HfDataset, keys: List[str], n: int = 10) -> int:
    assert isinstance(dataset, (Dataset, DatasetDict))
    dset = dataset if isinstance(dataset, Dataset) else next(iter(dataset.values()))
    sample = {k: v for k, v in dset[:n].items() if k in keys}
    batch_size = infer_batch_shape(sample)
    nesting_level = len(batch_size) - 1
    return nesting_level
