from __future__ import annotations

import datetime
import os
import shutil
import tempfile
from copy import copy
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import TypeVar

import datasets
import pytorch_lightning as pl
import rich
import tensorstore as ts
from datasets import Dataset
from datasets import DatasetDict
from datasets import Split
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch import nn

from warp_pipes.core.condition import Condition
from warp_pipes.core.condition import HasPrefix
from warp_pipes.core.condition import In
from warp_pipes.core.condition import Reduce
from warp_pipes.pipes import ApplyAsFlatten
from warp_pipes.pipes import Collate
from warp_pipes.pipes import Flatten
from warp_pipes.pipes import Pipe
from warp_pipes.pipes import Predict
from warp_pipes.pipes.predict import DEFAULT_LOADER_KWARGS
from warp_pipes.support.datasets_utils import get_dataset_fingerprints
from warp_pipes.support.datasets_utils import keep_only_columns
from warp_pipes.support.datastruct import Batch
from warp_pipes.support.datastruct import OutputFormat
from warp_pipes.support.datastruct import PathLike
from warp_pipes.support.fingerprint import get_fingerprint
from warp_pipes.support.search_engines import AutoEngine
from warp_pipes.support.search_engines import IndexEngine
from warp_pipes.support.shapes import infer_batch_shape

HfDataset = TypeVar("HfDataset", Dataset, DatasetDict)

ALLOWED_COLUMNS_NESTING = [
    "question.input_ids",
    "question.attention_mask",
    "question.text",
    "question.metamap",
    "question.answer_text",
    "document.row_idx",
    "document.proposal_score",
    "question.document_idx",
]
ALLOWED_COLUMNS_CACHING = [
    "question.input_ids",
    "question.attention_mask",
]
INFER_NESTING_KEYS = [
    "question.input_ids",
    "question.attention_mask",
    "question.text",
]

MAX_INDEX_CACHE_AGE = 60 * 60 * 24 * 3  # 3 days
TEMPDIR_SUFFIX = "-tempdir"


def cleanup_cache(cache_dir, max_age: int = MAX_INDEX_CACHE_AGE):
    if os.path.exists(cache_dir):
        for file in Path(cache_dir).iterdir():
            if file.is_dir() and file.name.endswith(TEMPDIR_SUFFIX):
                age = datetime.datetime.fromtimestamp(os.stat(file).st_ctime)
                time_diff = datetime.datetime.now() - age
                size = sum(f.stat().st_size for f in file.glob("**/*") if f.is_file())
                if time_diff.total_seconds() > max_age:
                    logger.warning(
                        f"Deleting cache {file} (age={time_diff}, size={size/2**30:.2f}GB)"
                    )
                    shutil.rmtree(file, ignore_errors=True)
                else:
                    logger.debug(
                        f"Keeping cache {file} (age={time_diff}, size={size/2**30:.2f}GB)"
                    )


def infer_nesting_level(dataset: HfDataset, keys: List[str], n: int = 10) -> int:
    assert isinstance(dataset, (Dataset, DatasetDict))
    dset = dataset if isinstance(dataset, Dataset) else next(iter(dataset.values()))
    sample = {k: v for k, v in dset[:n].items() if k in keys}
    batch_size = infer_batch_shape(sample)
    nesting_level = len(batch_size) - 1
    return nesting_level


class Index(Pipe):
    """Keep an index of a Dataset and search it using queries."""

    index_name: Optional[str] = None
    is_indexed: bool = False
    default_key: Optional[str | List[str]] = None
    no_fingerprint: List[str] = [
        "cache_dir",
        "trainer",
        "loader_kwargs",
    ]

    def __init__(
        self,
        corpus: Dataset,
        *,
        engines: List[IndexEngine | Dict] = None,
        query_field: str = "question",
        index_field: str = "document",
        model: pl.LightningModule | nn.Module = None,
        trainer: Optional[Trainer] = None,
        persist_cache: bool = False,
        cache_dir: PathLike = None,
        # Pipe args
        input_filter: Optional[Condition] = None,
        update: bool = False,
        # Argument for computing the vectors
        dtype: str = "float32",
        model_output_keys: List[str] = None,
        loader_kwargs: Optional[Dict] = None,
        corpus_collate_pipe: Pipe = None,
        dataset_collate_pipe: Pipe = None,
        **_,
    ):
        if cache_dir is None:
            cache_dir = datasets.cached_path("./index")

        if model_output_keys is None:
            model_output_keys = ["_hq_", "_hd_"]

        # clean the temporary cache
        cleanup_cache(cache_dir, max_age=MAX_INDEX_CACHE_AGE)

        # set the path where to store the index
        if not persist_cache:
            cache_dir = tempfile.mkdtemp(dir=cache_dir, suffix=TEMPDIR_SUFFIX)
        self.cache_dir = Path(cache_dir) / f"fz-index-{corpus._fingerprint}"

        # input fields and input filter for query time
        self.query_field = query_field
        self.index_field = index_field
        query_input_filter = HasPrefix(self.query_field)
        if input_filter is not None:
            query_input_filter = Reduce(query_input_filter, input_filter)

        # register the Engines
        if isinstance(engines, (dict, DictConfig)):
            assert all(isinstance(cfg, (dict, DictConfig)) for cfg in engines.values())
            engines = [{**{"name": k}, **v} for k, v in engines.items()]

        self.engines_config = copy(engines)
        self.engines = []
        for engine in engines:
            if isinstance(engine, dict):
                engine = AutoEngine(**engine, path=self.cache_dir, set_unique_path=True)
            elif isinstance(engine, IndexEngine):
                pass
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

        # build the Pipe
        super().__init__(
            input_filter=query_input_filter,
            update=update,
            id=type(self).__name__,
        )

        # Register the model and the pipes used
        # to handle the processing of the data
        assert dtype in {"float32", "float16"}
        self.dtype = dtype
        self.predict_docs = Predict(
            model, model_output_keys=model_output_keys, output_dtype=self.dtype
        )
        self.predict_queries = Predict(
            model, model_output_keys=model_output_keys, output_dtype=self.dtype
        )
        # trainer and dataloader
        self.trainer = trainer
        self.loader_kwargs = loader_kwargs or DEFAULT_LOADER_KWARGS
        # collate pipe use to convert dataset rows into a batch
        self.corpus_collate_pipe = corpus_collate_pipe or Collate()
        self.dataset_collate_pipe = dataset_collate_pipe or Collate()

        # build the engines and save them to disk
        self.build_engines(corpus)

    def build_engines(self, corpus):
        if self.requires_vector:
            vectors = self.cache_vectors(
                corpus,
                predict=self.predict_docs,
                trainer=self.trainer,
                collate_fn=self.corpus_collate_pipe,
                target_file=self.vector_file(self.predict_docs.model, suffix="corpus"),
                persist=True,
            )
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
            batch_ = self.dataset_collate_pipe(batch)
            vectors = self.predict_queries(batch_)[self.predict_queries.output_key]
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
        clean_caches: bool = True,
        set_new_fingerprint: bool = True,
        **kwargs,
    ) -> HfDataset:
        # infer the nesting level. I.e. Questions as shape [bs, ...] or [bs, n_options, ...]
        nesting_level = infer_nesting_level(dataset, INFER_NESTING_KEYS, n=10)

        # cache the query vectors
        if self.requires_vector:
            dataset_for_caching = keep_only_columns(dataset, ALLOWED_COLUMNS_CACHING)

            # flatten the dataset if it is nested
            if nesting_level > 0:
                flatten_op = Flatten(level=nesting_level)
                dataset_for_caching = flatten_op(
                    dataset_for_caching,
                    level=nesting_level,
                    keys=["question.input_ids", "question.attention_mask"],
                    desc="Flattening dataset before caching",
                    **kwargs,
                )

            # cache the vectors
            dset_fingerprint = get_dataset_fingerprints(
                dataset_for_caching, reduce=True
            )
            dset_cache_file = self.vector_file(
                self.predict_queries.model, suffix=f"dset-{dset_fingerprint}"
            )

            vectors = self.cache_vectors(
                dataset_for_caching,
                predict=self.predict_queries,
                trainer=self.trainer,
                collate_fn=self.dataset_collate_pipe,
                target_file=dset_cache_file,
                persist=True,
            )
            # put the model back to cpu to save memory
            self.predict_queries.model.cpu()
        else:
            vectors = None if isinstance(dataset, Dataset) else {}

        # fingerprint the state as: dataset_hash + \sum _i engine_i_hash, this allows
        # giving a unique fingerprint to the intermediate results of the engines
        fingerprint_state = get_dataset_fingerprints(dataset)

        # process the dataset with the Engines
        for engine in self.engines:
            # wrap the engine with a flattening op to handle the nesting level
            if nesting_level > 0:
                # Filter out the columns that cannot be flattened
                # TODO: this is a hack to avoid the problem of the engine
                #  not being able to handle the nesting level. Find a better way.
                input_filter = In(ALLOWED_COLUMNS_NESTING)

                # wrap the pipe with the flattening op
                pipe = ApplyAsFlatten(
                    engine,
                    level=nesting_level,
                    input_filter=input_filter,
                    flatten_idx=True,
                )
            else:
                pipe = engine

            # pipe = Sequential(PrintBatch(f"{engine.name}::input"), pipe)

            # process the dataset
            if isinstance(dataset, DatasetDict):
                dataset = DatasetDict(
                    {
                        split: pipe(
                            dset,
                            vectors=vectors.get(split, None)
                            if engine.require_vectors
                            else None,
                            output_format=OutputFormat.NUMPY,
                            set_new_fingerprint=set_new_fingerprint,
                            # fingerprint_state=fingerprint_state.get(split, None),
                            desc=f"{type(engine).__name__} ({split})",
                            **kwargs,
                        )
                        for split, dset in dataset.items()
                    }
                )
            elif isinstance(dataset, Dataset):
                dataset = pipe(
                    dataset,
                    vectors=vectors if engine.require_vectors else None,
                    output_format=OutputFormat.NUMPY,
                    set_new_fingerprint=set_new_fingerprint,
                    # fingerprint_state=fingerprint_state,
                    desc=f"{type(engine).__name__}",
                    **kwargs,
                )
            else:
                raise TypeError(f"Unsupported dataset type: {type(dataset)}")

            # free the CPU and GPU memory allocated by the engine
            engine.free_memory()

            # update the fingerprint state
            if isinstance(dataset, Dataset):
                fingerprint_state = get_fingerprint(
                    {"state": fingerprint_state, "new": engine.fingerprint(reduce=True)}
                )
            elif isinstance(dataset, DatasetDict):
                fingerprint_state = {
                    k: get_fingerprint(
                        {
                            "state": fingerprint_state,
                            "new": engine.fingerprint(reduce=True),
                        }
                    )
                    for k, v in fingerprint_state.items()
                }
            else:
                raise ValueError("Unsupported dataset type")

        # cleanup the vectors
        if clean_caches and self.requires_vector:
            self.predict_queries.delete_cached_files()
            self.predict_docs.delete_cached_files()

        return dataset

    def cache_vectors(
        self,
        dataset: Dataset | DatasetDict,
        *,
        predict: Predict,
        trainer: Trainer,
        collate_fn: Callable,
        cache_dir: Optional[Path] = None,
        target_file: Optional[PathLike] = None,
        **kwargs,
    ) -> ts.TensorStore | Dict[Split, ts.TensorStore]:
        predict.invalidate_cache()

        if isinstance(dataset, Dataset):
            predict.cache(
                dataset,
                trainer=trainer,
                collate_fn=collate_fn,
                loader_kwargs=self.loader_kwargs,
                cache_dir=cache_dir,
                target_file=target_file,
                **kwargs,
            )
            return self.read_vectors_table(target_file)
        elif isinstance(dataset, DatasetDict):
            vectors = {}
            for split, dset in dataset.items():
                target_file_split = target_file / str(split)
                predict.cache(
                    dset,
                    trainer=trainer,
                    collate_fn=collate_fn,
                    loader_kwargs=self.loader_kwargs,
                    cache_dir=cache_dir,
                    target_file=target_file_split,
                    **kwargs,
                )
                vectors[split] = self.read_vectors_table(target_file_split)

            return vectors
        else:
            raise TypeError(f"Unknown dataset type {type(dataset)}")

    def vector_file(self, model, suffix: str = "corpus"):
        model_fingerprint = get_fingerprint(model)
        path = (
            Path(self.cache_dir)
            / "vectors"
            / f"vectors-{model_fingerprint}-{suffix}.tsarrow"
        )
        return path

    def read_vectors_table(self, vector_file: Path) -> ts.TensorStore:
        logger.info(f"Reading vectors table from: {vector_file.absolute()}")
        return ts.TensorStore(vector_file, dtype=self.dtype)

    def __del__(self):
        if hasattr(self, "persist_cache") and not self.persist_cache:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    def __repr__(self):
        return f"{type(self).__name__}(engines={self.engines_config})"
