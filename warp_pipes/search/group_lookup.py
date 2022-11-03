from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import faiss.contrib.torch_utils  # type: ignore
import torch
from datasets import Dataset
from loguru import logger

from warp_pipes.search.base import Search
from warp_pipes.search.search_result import SearchResult
from warp_pipes.support.datastruct import Batch
from warp_pipes.support.tensor_handler import TensorLike


class GroupLookupSearch(Search):
    """Retrieve all the passages corresponding to a given group id."""

    _no_fingerprint: List[str] = Search._no_fingerprint + ["_lookup"]

    @property
    def index_group_key(self) -> str:
        return self.full_key(self.config.index_field, self.config.group_key)

    @property
    def query_group_key(self) -> str:
        return self.full_key(self.config.query_field, self.config.group_key)

    def _build(
        self,
        vectors: Optional[TensorLike] = None,
        corpus: Optional[Dataset] = None,
        **kwargs,
    ):
        """build the index from the vectors."""
        if corpus is None:
            raise ValueError("`corpus` is required")

        if self.index_group_key not in corpus.column_names:
            raise ValueError(
                f"`{self.index_group_key}` is "
                f"required to build the"
                f"{self.__class__.__name__}. "
                f"Found keys `{corpus.column_names}`"
            )
        doc_ids = corpus[self.index_group_key]

        # NB: for Colbert, this is a pretty ineffective way to build the index: this is an index
        # from document id to token id, an index from document id to passage id would be better.
        lookup_ = defaultdict(list)
        for tokid, doc_id in enumerate(doc_ids):
            lookup_[doc_id].append(tokid)

        n_cols = max(len(v) for v in lookup_.values()) + 1
        n_rows = max(lookup_.keys()) + 1

        self._lookup = torch.empty(n_rows, n_cols, dtype=torch.int64).fill_(-1)
        for i in range(n_rows):
            self._lookup[i, : len(lookup_[i])] = torch.tensor(sorted(lookup_[i]))

        logger.info(f"Lookup table: {self._lookup.shape}")
        self.save()

    def __len__(self) -> int:
        return self._lookup.shape[1]

    def lookup_file(self, savedir: Path) -> Path:
        return savedir / "lookup.pt"

    def _save_special_attrs(self, savedir: Path):
        torch.save(self._lookup, self.lookup_file(savedir))

    def _load_special_attrs(self, savedir: Path):
        self._lookup = torch.load(self.lookup_file(savedir))

    def cpu(self):
        """Move the index to CPU."""
        ...

    def cuda(self, devices: Optional[List[int]] = None):
        """Move the index to CUDA."""
        ...

    def free_memory(self):
        """Free the memory of the index."""
        self._lookup = None

    @property
    def is_up(self) -> bool:
        return self._lookup is not None

    def __del__(self):
        self.free_memory()

    def search(
        self, *query: Any, k: int = None, **kwargs
    ) -> Tuple[TensorLike, TensorLike]:
        doc_ids, *_ = query
        doc_ids = torch.tensor(doc_ids, dtype=torch.int64, device=self._lookup.device)
        pids = self._lookup[doc_ids]
        pids = pids[:, :k]
        scores = torch.zeros_like(pids, dtype=torch.float32)
        scores[pids < 0] = -math.inf
        return scores, pids

    def _search_chunk(self, query: Batch, *, k: int, **kwargs) -> SearchResult:
        if self.query_group_key not in query.keys():
            raise ValueError(
                f"`{self.query_group_key}` "
                f"is required. Found keys `{query.keys()}`."
            )

        doc_ids = query[self.query_group_key]
        scores, pids = self.search(doc_ids, k=k, **kwargs)
        return SearchResult(scores=scores, indices=pids)
