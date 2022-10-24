from __future__ import annotations

from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import faiss.contrib.torch_utils  # type: ignore
import numpy as np
import torch  # type: ignore
from datasets import Dataset
from loguru import logger

from warp_pipes.support.datastruct import Batch
from warp_pipes.support.search_engines.base import SearchEngine
from warp_pipes.support.search_engines.base import SearchEngineConfig
from warp_pipes.support.search_engines.search_result import SearchResult
from warp_pipes.support.search_engines.vector_base import VectorBase
from warp_pipes.support.search_engines.vector_base.auto import AutoVectorBase
from warp_pipes.support.search_engines.vector_base.base import VectorBaseConfig
from warp_pipes.support.tensor_handler import TensorLike


class DenseSearchEngineConfig(SearchEngineConfig, VectorBaseConfig):
    ...


class DenseSearchEngine(SearchEngine):
    """This class implements a low level index."""

    _config_type: type = DenseSearchEngineConfig
    _max_num_proc: int = 1
    require_vectors: bool = True

    def _build(
        self,
        vectors: Optional[TensorLike] = None,
        corpus: Optional[Dataset] = None,
        **kwargs,
    ):
        """build the index from the vectors."""
        assert (
            len(vectors.shape) == 2
        ), f"The vectors must be 2D. vectors: {vectors.shape}"
        self.config.dimension = vectors.shape[1]

        # init the index
        self._index = self._init_index(self.config)

        # train the index
        faiss_train_size = self.config.train_size
        random_train_subset = self.config.random_train_subset
        if faiss_train_size is not None and faiss_train_size < len(vectors):
            if random_train_subset:
                train_ids = np.random.choice(
                    len(vectors), faiss_train_size, replace=False
                )
            else:
                train_ids = slice(None, faiss_train_size)
        else:
            train_ids = slice(None, None)

        train_vectors = vectors[train_ids]
        logger.info(
            f"Train the index "
            f"with {len(train_vectors)} vectors "
            f"(type: {type(train_vectors)})."
        )
        self._index.train(train_vectors)

        # add vectors to the index
        logger.info(
            f"Adding {len(vectors)} vectors "
            f"(type: {type(vectors).__name__}, {vectors.shape}) to the index."
        )
        self._index.add(vectors)

        # make sure to free-up GPU memory
        self.cpu()

    def __len__(self) -> int:
        return self._index.ntotal

    def _init_index(self, config: VectorBaseConfig) -> VectorBase:
        import rich

        rich.print(config)
        return AutoVectorBase(config)

    def _save_special_attrs(self, savedir: Path):
        self._index.save(savedir)

    def _load_special_attrs(self, savedir: Path):
        self._index.load(savedir)

    def cpu(self):
        """Move the index to CPU."""
        self._index.cpu()

    def cuda(self, devices: Optional[List[int]] = None):
        """Move the index to CUDA."""
        self._index.cuda(devices)

    def free_memory(self):
        """Free the memory of the index."""
        self._index = None

    @property
    def is_up(self) -> bool:
        return self._index is not None

    def search(
        self, *query: TensorLike, k: int = None, **kwargs
    ) -> Tuple[TensorLike, TensorLike]:
        query_vectors, *_ = query
        return self._index.search(query_vectors, k=k)

    def _search_chunk(
        self, query: Batch, *, k: int, vectors: Optional[torch.Tensor], **kwargs
    ) -> SearchResult:
        scores, indices = self.search(vectors, k=k)
        return SearchResult(scores=scores, indices=indices)
