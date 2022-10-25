from __future__ import annotations

import logging
import math
from functools import partial
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

from datasets import Dataset
from elasticsearch import Elasticsearch
from loguru import logger
from tqdm.auto import tqdm

from warp_pipes.core.condition import In
from warp_pipes.pipes import RenameKeys
from warp_pipes.support.datasets_utils import keep_only_columns
from warp_pipes.support.datastruct import Batch
from warp_pipes.support.elasticsearch import es_create_index
from warp_pipes.support.elasticsearch import es_ingest
from warp_pipes.support.elasticsearch import es_remove_index
from warp_pipes.support.elasticsearch import es_search
from warp_pipes.support.search_engines.base import SearchEngine
from warp_pipes.support.search_engines.base import SearchEngineConfig
from warp_pipes.support.search_engines.search_result import SearchResult
from warp_pipes.support.tensor_handler import TensorFormat
from warp_pipes.support.tensor_handler import TensorLike


def _pad_to_length(lst: List, *, length: int, fill_token) -> List:
    if len(lst) < length:
        lst.extend([fill_token] * (length - len(lst)))
    return lst[:length]


class ElasticSearchEngineConfig(SearchEngineConfig):
    _no_fingerprint: List[str] = SearchEngineConfig._no_fingerprint + [
        "_instance",
        "timeout",
        "es_logging_level",
        "ingest_batch_size",
    ]
    _no_index_fingerprint = SearchEngineConfig._no_index_fingerprint + [
        "es_temperature",
        "auxiliary_weight" "scale_auxiliary_weight_by_lengths",
    ]

    es_index_key: str = "__ROW_IDX__"
    index_name: Optional[str] = None
    corpus_size: Optional[int] = None
    timeout: Optional[int] = 60
    es_body: Optional[Dict] = None
    main_key: str = "text"
    auxiliary_key: str = "title"
    filter_key: Optional[str] = None
    ingest_batch_size: int = 1000
    auxiliary_weight: float = 0
    scale_auxiliary_weight_by_lengths: bool = True
    es_temperature: float = 1.0
    es_logging_level: str = "error"


class ElasticSearchEngine(SearchEngine):
    _max_num_proc: int = None
    _instance: Elasticsearch
    _config_type = ElasticSearchEngineConfig

    @property
    def input_keys(self):
        keys = [self.config.main_key, self.config.auxiliary_key, self.config.filter_key]
        keys = filter(None, keys)
        return list(keys)

    @property
    def index_columns(self):
        return [self.full_key(self.config.index_field, k) for k in self.input_keys]

    @property
    def query_columns(self):
        return [self.full_key(self.config.query_field, k) for k in self.input_keys]

    def _build(
        self,
        vectors: Optional[TensorLike] = None,
        corpus: Optional[Dataset] = None,
        **kwargs,
    ):

        if corpus is None:
            raise ValueError("The corpus is required.")

        # keep only the relevant columns
        corpus = keep_only_columns(corpus, self.index_columns)

        # set a unique index name
        self.config.index_name = self._get_index_name(corpus, self.config)
        self.config.corpus_size = len(corpus)

        # instantiate the ElasticSearch instance
        self._init_es_instance()

        # init the index
        is_new_index = es_create_index(
            self.config.index_name, body=self.config.es_body, es_instance=self.instance
        )
        if not is_new_index:
            logger.info(
                f"ElasticSearch index with name=`{self.config.index_name}` already exists."
            )

        # build the index
        if is_new_index:
            batch_size = self.config.ingest_batch_size
            try:
                for i in tqdm(
                    range(0, len(corpus), batch_size), desc="Ingesting ES index"
                ):
                    batch = corpus[i : i + batch_size]
                    batch[self.config.es_index_key] = list(
                        range(i, min(len(corpus), i + batch_size))
                    )

                    _ = es_ingest(
                        batch,
                        index_name=self.config.index_name,
                        es_instance=self.instance,
                    )
            except Exception as ex:
                # clean up the index if something went wrong
                es_remove_index(self.config.index_name, es_instance=self.instance)
                raise ex

    def _init_es_instance(self):
        log_level = {
            "error": logging.ERROR,
            "warning": logging.WARNING,
            "info": logging.INFO,
            "debug": logging.DEBUG,
        }[self.config.es_logging_level]
        logging.getLogger("elasticsearch").setLevel(log_level)
        self._instance = Elasticsearch(timeout=self.config.timeout)

    def _load_special_attrs(self, savedir: Path):
        self._init_es_instance()

    def _save_special_attrs(self, savedir: Path):
        ...

    @property
    def instance(self):
        return self._instance

    def rm(self):
        """Remove the index."""
        es_remove_index(self.config.index_name, es_instance=self.instance)

    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        return -1

    def cpu(self):
        """Move the index to CPU."""
        pass

    def cuda(self, devices: Optional[List[int]] = None):
        """Move the index to CUDA."""
        pass

    def free_memory(self):
        """Free the memory of the index."""
        pass

    @property
    def is_up(self) -> bool:
        """Check if the index is up."""
        is_new_index = es_create_index(
            self.config.index_name, es_instance=self.instance
        )
        instance_init = getattr(self, "_instance")
        return instance_init is not None and not is_new_index

    def search(self, *query: Batch, k: int = None, **kwargs) -> SearchResult:
        """Search the index for a query and return the top-k results."""
        k = k or self.config.k
        rename_input_fields = RenameKeys(
            {
                f"{self.config.query_field}.{key}": f"{self.config.index_field}.{key}"
                for key in self.input_keys
            },
            input_filter=In(self.query_columns),
        )

        # unpack args and preprocess
        query, *_ = query
        query = rename_input_fields(query)

        # query Elastic Search
        output = es_search(
            query,
            k=k,
            auxiliary_weight=self.config.auxiliary_weight,
            query_key=self.full_key(self.config.index_field, self.config.main_key),
            auxiliary_key=self.full_key(
                self.config.index_field, self.config.auxiliary_key
            ),
            filter_key=self.full_key(self.config.index_field, self.config.filter_key),
            scale_auxiliary_weight_by_lengths=self.config.scale_auxiliary_weight_by_lengths,
            index_name=self.config.index_name,
            es_instance=self.instance,
        )
        scores = output["scores"]
        indices = output[self.config.es_index_key]
        if self.config.es_temperature is not None:
            scores = [
                [s / self.config.es_temperature for s in s_list] for s_list in scores
            ]

        # pad the scores and indices
        k_batch_max = max(len(s) for s in scores)
        pad_fn_scores = partial(
            _pad_to_length, fill_token=-math.inf, length=k_batch_max
        )
        scores = list(map(pad_fn_scores, scores))
        pad_fn_indices = partial(_pad_to_length, fill_token=-1, length=k_batch_max)
        indices = list(map(pad_fn_indices, indices))

        # build the results
        return SearchResult(scores=scores, indices=indices, format=TensorFormat.NUMPY)

    @staticmethod
    def _get_value(batch: Dict, column: List | str):
        if isinstance(column, str):
            column = [column]
        for c in column:
            if c in batch:
                return batch[c]

        return None

    def _search_chunk(
        self, batch: Batch, idx: Optional[List[int]] = None, **kwargs
    ) -> SearchResult:
        search_result = self.search(batch, **kwargs)
        return search_result

    def __getstate__(self):
        """this method is called when attempting pickling.
        ES instances cannot be properly pickled"""
        state = self.__dict__.copy()
        # Don't pickle the ES instances
        for attr in ["_instance"]:
            if attr in state:
                state.pop(attr)

        return state

    def __setstate__(self, state):
        self.__dict__ = state
        state["_instance"] = Elasticsearch(timeout=state["config"]["timeout"])
