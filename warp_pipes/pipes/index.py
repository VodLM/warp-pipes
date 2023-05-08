from __future__ import annotations

import os
import tempfile
from copy import copy
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import omegaconf
import pytorch_lightning as pl
from datasets import Dataset
from datasets import DatasetDict
from loguru import logger
from omegaconf import DictConfig
from torch import nn

from warp_pipes.core.pipe import Pipe
from warp_pipes.pipes import predict
from warp_pipes.search import AutoSearchEngine
from warp_pipes.search import Search
from warp_pipes.search.auto import AutoSearchConfig
from warp_pipes.support import caching
from warp_pipes.support.datasets_utils import HfDataset
from warp_pipes.support.datastruct import Batch
from warp_pipes.support.shapes import infer_batch_shape

TEMPDIR_SUFFIX = "-tempdir"


def _get_unique(x: List):
    y = list(set(x))
    assert len(y) == 1
    return y[0]


class Index(Pipe):
    """Keep an index of a Dataset and search it using queries."""

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
        engines: List[Search | Dict] = None,
        trainer: pl.Trainer = None,
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
            engines = [v for k, v in engines.items()]

        self.engines_config = copy(engines)
        self.engines = []
        for engine in engines:
            if isinstance(engine, (dict, DictConfig)):
                engine_config = AutoSearchConfig(
                    name=engine["name"], config=engine["config"]
                )
            elif isinstance(engine, Search):
                engine_config = engine.config
            else:
                raise ValueError(
                    f"Unknown engine type: {type(engine)} (accepted: dict, Search)"
                )

            # set the engine path
            engine_path = cache_dir / f"search-{engine_config.fingerprint}"

            if isinstance(engine, (dict, DictConfig)):
                engine = AutoSearchEngine(
                    name=engine["name"], path=engine_path, config=engine_config
                )
            elif isinstance(engine, Search):
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

        # validate the engines
        self._validate_engines(self.engines)

        # Register the model and the pipes used
        # to handle the processing of the data
        print(trainer)
        self.predict_index = predict.Predict(
            model, trainer=trainer, cache_dir=cache_dir, cache_config=index_cache_config
        )
        self.predict_queries = predict.Predict(
            model, trainer=trainer, cache_dir=cache_dir, cache_config=query_cache_config
        )

        # build the engines and save them to disk
        self.build_engines(corpus)

    def _validate_engines(self, engines: List[Search]):
        if len(engines) == 0:
            raise ValueError("No engines were registered, engine list is empty.")

        # check that all query fields are identical
        all_query_fields = [e.config.query_field for e in engines]
        if len(set(all_query_fields)) != 1:
            raise ValueError(
                f"`query_field` do not match, "
                f"found {len(set(all_query_fields))} values: "
                f"{all_query_fields}"
            )

        # check that all index fields are identical
        all_index_fields = [e.config.index_field for e in engines]
        if len(set(all_index_fields)) != 1:
            raise ValueError(
                f"`index_field` do not match, "
                f"found {len(set(all_index_fields))} values: "
                f"{all_index_fields}"
            )

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
        kwargs["set_new_fingerprint"] = True
        kwargs["print_fringerprint_dict"] = False
        if "desc" in kwargs:
            desc_ = f': {kwargs.pop("desc")}'
        else:
            desc_ = ""
        for engine in self.engines:
            if isinstance(dataset, DatasetDict):
                dataset = DatasetDict(
                    {
                        split: engine(
                            dset,
                            vectors=vectors.get(split, None)
                            if engine.require_vectors
                            else None,
                            desc=f"{type(engine).__name__} ({split}){desc_}",
                            **kwargs,
                        )
                        for split, dset in dataset.items()
                    }
                )
            elif isinstance(dataset, Dataset):
                dataset = engine(
                    dataset,
                    vectors=vectors if engine.require_vectors else None,
                    desc=f"{type(engine).__name__}{desc_}",
                    **kwargs,
                )
            else:
                raise TypeError(f"Unsupported dataset type: {type(dataset)}")

            # free the CPU and GPU memory allocated by the engine
            engine.free_memory()

        return dataset

    def __repr__(self):
        return (
            f"{type(self).__name__}(engines={[type(e).__name__ for e in self.engines]})"
        )

    @classmethod
    def instantiate_test(cls, cache_dir: Path, **kwargs) -> "Search":
        # TODO: test this
        return None


def infer_nesting_level(dataset: HfDataset, keys: List[str], n: int = 10) -> int:
    assert isinstance(dataset, (Dataset, DatasetDict))
    dset = dataset if isinstance(dataset, Dataset) else next(iter(dataset.values()))
    sample = {k: v for k, v in dset[:n].items() if k in keys}
    batch_size = infer_batch_shape(sample)
    nesting_level = len(batch_size) - 1
    return nesting_level
