from __future__ import annotations

import abc
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import faiss.contrib.torch_utils  # type: ignore
import rich
import torch
from datasets import Dataset
from datasets.search import SearchResults
from fz_openqa.datamodules.index.engines.base import IndexEngine
from fz_openqa.datamodules.index.engines.vector_base.utils.faiss import TensorLike
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.tensor_arrow import TensorArrowTable
from loguru import logger


class DocumentLookupEngine(IndexEngine):
    """Retrieve all the passages corresponding to a given document id."""

    _no_fingerprint: List[str] = IndexEngine.no_fingerprint + ["_lookup"]

    def _build(
        self,
        vectors: Optional[TensorLike | TensorArrowTable] = None,
        corpus: Optional[Dataset] = None,
    ):
        """build the index from the vectors."""
        if corpus is None:
            raise ValueError("`corpus` is required")

        if self.corpus_document_idx_key not in corpus.column_names:
            raise ValueError(
                f"`{self.corpus_document_idx_key}` is "
                f"required to build the"
                f"{self.__class__.__name__}. "
                f"Found {corpus.column_names}."
            )
        doc_ids = corpus[self.corpus_document_idx_key]

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

    @property
    def lookup_file(self) -> Path:
        return self.path / "lookup.pt"

    def save(self):
        """save the index to file"""
        super().save()
        torch.save(self._lookup, self.lookup_file)

    def load(self):
        """save the index to file"""
        super().load()
        self._lookup = torch.load(self.lookup_file)

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

    def search(self, *query: Any, k: int = None, **kwargs) -> SearchResult:
        doc_ids, *_ = query
        doc_ids = torch.tensor(doc_ids, dtype=torch.int64, device=self._lookup.device)
        pids = self._lookup[doc_ids][:, :k]
        scores = torch.zeros_like(pids, dtype=torch.float32)
        return SearchResult(score=scores, index=pids, k=k)

    def _search_chunk(
        self, query: Batch, *, k: int, vectors: Optional[torch.Tensor], **kwargs
    ) -> SearchResult:
        if self.dataset_document_idx_key not in query.keys():
            raise ValueError(
                f"`{self.dataset_document_idx_key}` "
                f"is required. Found {query.keys()}."
            )

        doc_ids = query[self.dataset_document_idx_key]
        return self.search(doc_ids, k=k, vectors=vectors, **kwargs)
