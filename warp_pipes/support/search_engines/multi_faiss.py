from __future__ import annotations

import json
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import faiss.contrib.torch_utils  # type: ignore
import numpy as np
import torch
from fz_openqa.datamodules.index.engines.base import IndexEngine
from fz_openqa.datamodules.index.engines.faiss import FaissEngine
from fz_openqa.utils.datastruct import PathLike
from fz_openqa.utils.tensor_arrow import TensorArrowTable
from tqdm import tqdm


class MultiFaissHandler(IndexEngine):
    """
    This class handles a faiss index for each Document.

    NB: this problem should be handled by the MaxSim ranker directly:
    in that case, this would be more effective to create the partitions of
    indices based on the document ids. This is not the case here, this module
    handles a different partition. This is a temporary workaround
    """

    _max_num_proc: int = 1

    def _build(
        self,
        vectors: torch.Tensor | TensorArrowTable | np.ndarray,
        *,
        doc_ids: Optional[List[int]] = None,
        **kwargs,
    ):
        """build the index from the vectors."""
        if doc_ids is None:
            raise ValueError(
                "Document ids must be provided via the argument `doc_ids`."
            )
        if isinstance(vectors, TensorArrowTable):
            vectors = vectors[:]
        elif isinstance(vectors, np.ndarray):
            vectors = torch.from_numpy(vectors)

        assert vectors.dim() == 2, f"The vectors must be 2D. vectors: {vectors.shape}"
        if not len(vectors) == len(doc_ids):
            raise ValueError(
                f"The length of vectors and doc_ids must be equal. "
                f"Found: {len(vectors)} != {len(doc_ids)}. "
                f"vectors: shape={vectors.shape} "
                f"({type(vectors).__name__})."
            )

        self.index_map = self.get_indexes_map(doc_ids)
        self.indexes = {}
        self.id_lookup = {}
        for doc_id, doc_path in tqdm(self.index_map.items(), desc="Building indexes.."):

            index = FaissEngine(path=doc_path, **kwargs)
            ids = [i for i, d in enumerate(doc_ids) if d == doc_id]
            if len(ids) == 0:
                raise ValueError(f"No vectors found for doc_id: {doc_id}")

            index.build(vectors[ids])
            self.indexes[doc_id] = index
            self.id_lookup[doc_id] = torch.tensor(ids)

    def get_indexes_map(self, doc_ids: List[int]) -> Dict[int, str]:
        return {
            int(doc_id): str(self.path / f"{doc_id}-doc-index")
            for doc_id in set(doc_ids)
        }

    def __len__(self) -> int:
        return sum(len(index) for index in self.indexes.values())

    @property
    def index_map_path(self) -> PathLike:
        return self.path / "index_map.json"

    @property
    def id_lookup_path(self) -> PathLike:
        return self.path / "id_lookup.json"

    def save(self):
        """save the index to file"""
        super().save()
        with open(str(self.index_map_path), "w") as f:
            f.write(json.dumps(self.index_map))
        with open(str(self.id_lookup_path), "w") as f:
            id_lookup = {k: v.tolist() for k, v in self.id_lookup.items()}
            f.write(json.dumps(id_lookup))
        for index in self.indexes.values():
            index.save()

    def load(self):
        """save the index to file"""
        super().load()
        with open(str(self.index_map_path), "r") as f:
            self.index_map = json.loads(f.read())
        with open(str(self.id_lookup_path), "r") as f:
            id_lookup = json.loads(f.read())
            self.id_lookup = {int(k): torch.tensor(v) for k, v in id_lookup.items()}
        self.indexes = {}
        for doc_id, doc_path in self.index_map.items():
            self.indexes[int(doc_id)] = IndexEngine.load_from_path(path=doc_path)

        assert set(self.id_lookup.keys()) == set(self.indexes.keys())

    def cpu(self):
        """Move the index to CPU."""
        for index in self.indexes.values():
            index.cpu()

    def cuda(self, devices: Optional[List[int]] = None):
        """Move the index to CUDA."""
        for index in self.indexes.values():
            index.cuda(devices)

    def free_memory(self):
        """Free the memory of the index."""
        for index in self.indexes.values():
            index.free_memory()

    @property
    def is_up(self) -> bool:
        return all(index.is_up for index in self.indexes.values())

    def __del__(self):
        self.free_memory()

    def __call__(
        self,
        query: torch.Tensor,
        *,
        k: int,
        doc_ids: List[int] | torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call the index."""
        udoc_ids = (
            set(doc_ids)
            if isinstance(doc_ids, list)
            else set(doc_ids.unique().tolist())
        )
        if doc_ids is None:
            raise ValueError("doc_ids must be provided.")
        if not len(query) == len(doc_ids):
            raise ValueError(
                f"The length of query and doc_ids must be equal. "
                f"Found: {len(query)} != {len(doc_ids)}. "
            )
        if not udoc_ids <= set(self.indexes.keys()):
            raise ValueError(
                f"The doc_ids must be a subset of the keys of the index. "
                f"Found: {set(doc_ids)} != {set(self.indexes.keys())}."
            )

        # split the data
        scores = []
        indexes = []
        for doc_idx in udoc_ids:
            ids = [i for i, did in enumerate(doc_ids) if did == doc_idx]
            if len(ids) == 0:
                continue

            # query the index
            index_doc = self.indexes[doc_idx]
            scores_doc, indexes_doc = index_doc(query[ids], k=k)

            # lookup the original ids
            lookup = self.id_lookup[doc_idx]
            indexes_doc = lookup[indexes_doc]

            for i, scores_i, indexes_i in zip(ids, scores_doc, indexes_doc):
                scores.append((i, scores_i))
                indexes.append((i, indexes_i))

        scores = self._concatenate(scores)
        indexes = self._concatenate(indexes)

        if not len(scores) == len(query):
            raise ValueError(
                f"The length of scores and query must be equal. "
                f"Found: {len(scores)} != {len(query)}."
            )

        return scores, indexes

    @staticmethod
    def _concatenate(values: List[Tuple[int, torch.Tensor]]) -> torch.Tensor:
        values = sorted(values, key=lambda x: x[0])
        values = torch.cat([x[None] for i, x in values])
        return values
