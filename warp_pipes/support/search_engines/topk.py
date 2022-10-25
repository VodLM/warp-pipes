from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import List
from typing import Optional

import faiss.contrib.torch_utils  # type: ignore
import torch  # type: ignore
from datasets import Dataset
from loguru import logger

from warp_pipes.support.datastruct import Batch
from warp_pipes.support.search_engines.base import SearchEngine
from warp_pipes.support.search_engines.search_result import SearchResult
from warp_pipes.support.tensor_handler import TensorFormat
from warp_pipes.support.tensor_handler import TensorHandler
from warp_pipes.support.tensor_handler import TensorLike


class TopkSearchEngine(SearchEngine):
    _max_num_proc: int = None
    require_vectors: bool = False

    def _build(
        self,
        vectors: Optional[TensorLike] = None,
        corpus: Optional[Dataset] = None,
        **kwargs,
    ):
        # check and cast the inputs
        if self.config.merge_previous_results:
            logger.warning(
                "merge_previous_results is set to True, "
                "but MaxSimEngine is a re-ranker: setting "
                "merge_previous_results to False"
            )
            self.config.merge_previous_results = False

    def cpu(self):
        pass

    def cuda(self, devices: Optional[List[int]] = None):
        pass

    def _save_special_attrs(self, savedir: Path):
        pass

    def _load_special_attrs(self, savedir: Path):
        pass

    @property
    def is_up(self) -> bool:
        return True

    def free_memory(self):
        pass

    def search(self, *query: Any, k: int = None, **kwargs) -> (TensorLike, TensorLike):
        # unpack and convert inputs
        scores, pids, *_ = query
        handler = TensorHandler(TensorFormat.TORCH)
        scores = handler(scores)
        pids = handler(pids)

        # sort and truncate
        idx = torch.argsort(scores, descending=True, dim=1)[:, :k]
        scores = scores.gather(1, index=idx)
        pids = pids.gather(1, index=idx)
        return scores, pids

    def _search_chunk(
        self,
        query: Batch,
        *,
        k: int,
        vectors: Optional[torch.Tensor],
        scores: Optional[torch.Tensor] = None,
        pids: Optional[TensorLike] = None,
        **kwargs,
    ) -> SearchResult:
        scores, indices = self.search(scores, pids, k=k)

        # here, fill_missing_values=True so -1 indices are filled with random indices
        return SearchResult(
            scores=scores,
            indices=indices,
        )
