from __future__ import annotations

import abc
import json
from copy import copy
from os import PathLike
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import omegaconf
import torch
from datasets import Dataset
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from omegaconf import ListConfig
from pydantic import BaseModel

from warp_pipes.core.fingerprintable import Fingerprintable
from warp_pipes.core.pipe import Pipe
from warp_pipes.support.datastruct import Batch
from warp_pipes.support.fingerprint import get_fingerprint
from warp_pipes.support.functional import camel_to_snake
from warp_pipes.support.functional import get_batch_eg
from warp_pipes.support.pretty import pprint_batch
from warp_pipes.support.search_engines.search_result import SearchResult
from warp_pipes.support.shapes import infer_batch_size
from warp_pipes.support.tensor_handler import TensorFormat
from warp_pipes.support.tensor_handler import TensorHandler
from warp_pipes.support.tensor_handler import TensorLike


def _stack_nested_tensors(index):
    """Transform nested Tensors into a single Tensor. I.e. [Tensor, Tensor, ...] -> Tensor"""
    if isinstance(index, list) and isinstance(index[0], (np.ndarray, torch.Tensor)):
        if isinstance(index[0], np.ndarray):
            index = np.stack(index)
        elif isinstance(index[0], torch.Tensor):
            index = torch.stack(index)
        else:
            raise TypeError(f"Unsupported type: {type(index[0])}")
    return index


class SearchEngineConfig(BaseModel, Fingerprintable):
    """Base class for search engine configuration."""

    _no_fingerprint: List[str] = ["path", "max_batch_size", "verbose"]
    _no_corpus_fingerprint: List[str] = []
    # main arguments
    path: Path
    k: int = 10
    merge_previous_results: bool = False
    require_vectors: bool = False
    k_max: Optional[int] = None
    # name of the query and corpus fields
    query_field = "question"
    output_score_key = "document.proposal_score"
    output_index_key = "document.row_idx"
    corpus_document_idx_key = "document.idx"
    dataset_document_idx_key = "question.document_idx"
    index_keys: List[str] = []
    query_keys: List[str] = []
    # arguments
    max_batch_size: Optional[int] = 100
    verbose: bool = False

    def get_fingerprint(self, reduce: bool = False) -> str | Dict[str, Any]:
        """Fingerprints the arguments used at query time."""
        fingerprints = copy(self.__dict__)
        for exclude in self._no_fingerprint:
            fingerprints.pop(exclude, None)
        if reduce:
            return get_fingerprint(fingerprints)
        else:
            return fingerprints

    def get_indexing_fingerprint(self, reduce: bool = False) -> str:
        """Fingerprints the arguments used at indexing time."""
        fingerprints = self.get_fingerprint(reduce=False)
        for exclude in self._no_corpus_fingerprint:
            fingerprints.pop(exclude, None)
        if reduce:
            return get_fingerprint(fingerprints)
        else:
            return fingerprints


class SearchEngine(Pipe, metaclass=abc.ABCMeta):
    """This class implements an index."""

    no_fingerprint: List[str] = []
    no_index_name: List[str] = []
    _config_type: type = SearchEngineConfig

    def __init__(
        self,
        config: SearchEngineConfig | Dict | DictConfig,
        *,
        # Pipe args
        input_filter: None = None,
        update: bool = False,
    ):
        super().__init__(input_filter=input_filter, update=update)
        self.config = self._parse_config(config)

    def _parse_config(self, config: Dict | DictConfig) -> SearchEngineConfig:
        """Parse the configuration."""
        if isinstance(config, DictConfig):
            config = omegaconf.OmegaConf.to_container(config)
        if isinstance(config, dict):
            config = SearchEngineConfig(**config)
        if not isinstance(config, SearchEngineConfig):
            raise TypeError(
                f"Unsupported type: {type(config)} (Expected: {SearchEngineConfig})"
            )
        return config

    def save(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.state_file), "w") as f:
            f.write(json.dumps(self._get_state()))
        self._save_special_attrs(self.config.path)

    def load(self):
        with open(str(self.state_file), "r") as f:
            state = json.load(f)
            self.config = self._parse_config(state.pop("config"))
        self._load_special_attrs(self.config.path)

    @property
    def state_file(self) -> Path:
        return Path(self.config.path) / "state.json"

    def _get_state(self) -> Dict[str, Any]:
        state = {}
        state["config"] = self.config.json()
        state["_target_"] = type(self).__module__ + "." + type(self).__qualname__
        return state

    def build(
        self,
        *,
        vectors: Optional[TensorLike] = None,
        corpus: Optional[Dataset] = None,
    ):
        if self.exists():
            logger.info(f"Loading index from {self.config.path}")
            self.load()
        else:
            logger.info(f"Creating index at {self.config.path}")
            if self.config.require_vectors and vectors is None:
                raise ValueError(
                    f"{self.name} requires vectors, but none were provided"
                )
            self._build(vectors=vectors, corpus=corpus)
            self.save()
            if not self.exists():
                raise ValueError(f"Index {type(self).__name__} was not created.")

    def rm(self):
        """Remove the index."""
        if self.config.path.exists():
            if self.config.path.is_dir():
                self.config.path.rmdir()
            else:
                self.config.path.unlink()

    def exists(self):
        """Check if the index exists."""
        return self.config.path.exists()

    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        return -1

    @abc.abstractmethod
    def _build(
        self,
        vectors: Optional[TensorLike] = None,
        corpus: Optional[Dataset] = None,
        **kwargs,
    ):
        """build the index from the vectors or text."""
        ...

    @abc.abstractmethod
    def _save_special_attrs(self, savedir: Path):
        """save the attributes specific to the sub-class."""
        ...

    @abc.abstractmethod
    def _load_special_attrs(self, savedir: Path):
        """load the attributes specific to the sub-class."""
        ...

    @abc.abstractmethod
    def cpu(self):
        """Move the index to CPU."""
        ...

    @abc.abstractmethod
    def cuda(self, devices: Optional[List[int]] = None):
        """Move the index to CUDA."""
        ...

    @abc.abstractmethod
    def free_memory(self):
        """Free the memory occupied by the index."""
        ...

    @property
    @abc.abstractmethod
    def is_up(self) -> bool:
        """Check if the index is up."""
        ...

    @abc.abstractmethod
    def search(
        self, *query: Any, k: int = None, **kwargs
    ) -> Tuple[TensorLike, TensorLike]:
        ...

    @abc.abstractmethod
    def _search_chunk(
        self, query: Batch, *, k: int, vectors: Optional[torch.Tensor], **kwargs
    ) -> SearchResult:
        ...

    def _call_batch(
        self,
        query: Batch,
        idx: Optional[List[int]] = None,
        k: Optional[int] = None,
        vectors: Optional[TensorLike] = None,
        format: TensorFormat = TensorFormat.NUMPY,
        **kwargs,
    ) -> Batch:
        """
        Search the index for a batch of examples (query).

        Filter the incoming batch using the same pipe as the one
        used to build the index.

        # TODO: keep `vectors` in the batch
        """
        k = k or self.config.k
        pprint_batch(
            query, f"{type(self).__name__}::base::query", silent=not self.config.verbose
        )

        # Auto-load the engine if it is not already done.
        if not self.is_up:
            self.load()
            self.cuda()
            assert self.is_up, f"Index {type(self).__name__} is not up."

        # get the indices given by the previous engine, if any
        prev_search_results = None
        if self.config.output_index_key in query:
            indices = _stack_nested_tensors(query[self.config.output_index_key])
            scores = _stack_nested_tensors(query[self.config.output_score_key])
            if (
                self.config.output_index_key in query
                and self.config.merge_previous_results
            ):
                prev_search_results = SearchResult(
                    index=indices,
                    score=scores,
                    format=format,
                )
            if isinstance(indices, np.ndarray):
                indices = torch.from_numpy(indices)
            if isinstance(scores, np.ndarray):
                scores = torch.from_numpy(scores)

        # fetch the query vectors as Tensors
        if vectors is None:
            q_vectors = None
        else:
            q_vectors = TensorHandler(TensorFormat.TORCH)(vectors, key=idx)

        # search the index by chunk
        batch_size = infer_batch_size(query)
        search_results = None
        if self.config.max_batch_size is not None:
            eff_batch_size = min(max(1, self.config.max_batch_size), batch_size)
        else:
            eff_batch_size = batch_size
        for i in range(0, batch_size, eff_batch_size):

            # slice the query
            chunk_i = get_batch_eg(query, slice(i, i + eff_batch_size))
            if q_vectors is not None:
                q_vectors_i = q_vectors[i : i + eff_batch_size]
            else:
                q_vectors_i = None
            if prev_search_results is not None:
                indices_i = prev_search_results.indices[i : i + eff_batch_size]
                scores_i = prev_search_results.scores[i : i + eff_batch_size]
            else:
                indices_i = scores_i = None

            # search the index for the chunk
            r = self._search_chunk(
                chunk_i,
                k=k,
                vectors=q_vectors_i,
                indices=indices_i,
                scores=scores_i,
                **kwargs,
            ).to(format)

            # potentially resize the results
            if self.config.k_max is not None:
                if r.shape[1] > self.config.k_max:
                    r = r.resize(self.config.k_max)

            # append the batch of search results to the previous ones
            if search_results is None:
                search_results = r
            else:
                search_results = search_results.append(r)

        # merge with the previous results
        if (
            prev_search_results is not None
            and self.config.merge_previous_results is False
        ):
            search_results = search_results + prev_search_results

        # format the output
        search_results = search_results.to(format=format)
        output = {
            self.config.output_index_key: search_results.indices,
            self.config.output_score_key: search_results.scores,
        }

        pprint_batch(
            output,
            f"{type(self).__name__}::base::output",
            silent=not self.config.verbose,
        )

        return output

    def _get_index_name(self, dataset, config) -> str:
        """Set the index name. Must be unique to allow for safe caching."""
        cls_id = camel_to_snake(type(self).__name__)
        index_cfg = self.config.get_indexing_fingerprint(reduce=True)
        return f"{cls_id}-{dataset._fingerprint}-{index_cfg}"

    def __del__(self):
        self.free_memory()

    @staticmethod
    def cast_attr(x: Any):
        if isinstance(x, (ListConfig, DictConfig)):
            return omegaconf.OmegaConf.to_container(x)
        return x

    @property
    def name(self) -> str:
        return type(self).__name__

    @classmethod
    def load_from_path(cls, path: PathLike):
        state_path = Path(path) / "state.json"
        with open(state_path, "r") as f:
            config = json.load(f)
            instance = instantiate(config)
            instance.load()

        return instance
