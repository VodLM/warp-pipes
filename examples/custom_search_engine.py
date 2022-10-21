from __future__ import annotations

import abc
import json
from copy import copy
from pathlib import Path
from tempfile import tempdir
import tempfile
import shutil
from typing import Any
from typing import Dict
from typing import List
from typing import Optional, Tuple

from os import PathLike
import numpy as np
import omegaconf
import torch
from datasets import Dataset
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from omegaconf import ListConfig

from warp_pipes.support.search_engines.search_result import SearchResult
from warp_pipes.support.functional import camel_to_snake
from warp_pipes.support.functional import get_batch_eg
from warp_pipes.core.pipe import Pipe
from warp_pipes.support.datastruct import Batch
from warp_pipes.support.tensor_handler import TensorFormat, TensorLike, TensorHandler
from warp_pipes.support.fingerprint import get_fingerprint
from warp_pipes.support.shapes import infer_batch_size
from warp_pipes.support.pretty import pprint_batch

from random import random
from warp_pipes.support.search_engines.base import SearchEngine
import numpy as np
import string
import datasets
import rich
from pathlib import Path


class CustomEngine(SearchEngine):

    def _build(self, vectors: Optional[TensorLike] = None,
        corpus: Optional[Dataset] = None, **kwargs):
        """build the index from the vectors or text."""
        self.vectors = TensorHandler(TensorFormat.TORCH)(vectors)

    def _save_special_attrs(self, savedir: Path):
        """save the attributes specific to the sub-class."""
        torch.save(self.vectors, savedir / "vectors.pt")

    def _load_special_attrs(self, savedir: Path):
        """load the attributes specific to the sub-class."""
        self.vectors = torch.load(savedir / "vectors.pt")

    def cpu(self):
        """Move the index to CPU."""
        ...

    def cuda(self, devices: Optional[List[int]] = None):
        """Move the index to CUDA."""
        ...

    def free_memory(self):
        """Free the memory occupied by the index."""
        ...

    @property
    def is_up(self) -> bool:
        """Check if the index is up."""
        return True

    def search(self, query: TensorLike, k: int = 10, **kwargs) -> Tuple[TensorLike, TensorLike]:
        query = TensorHandler(TensorFormat.TORCH)(query)
        sim = query @ self.vectors.T
        indices = torch.argsort(sim, descending=True, dim=1)[:, :k]
        return sim.gather(1, index=indices), indices

    def _search_chunk(
        self, query: Batch, *, k: int, vectors: Optional[torch.Tensor], **kwargs
    ) -> SearchResult:
        """Search the index for the given query."""
        scores, indices = self.search(vectors, k=k)
        return SearchResult(scores=scores, indices=indices)


def run(cache_dir):
    # make dataset with random strings and random vectors
    vectors = np.random.randn(100, 8).astype(np.float32)
    data = ["".join(np.random.choice(list(string.ascii_letters), 10).tolist() ) for _ in range(100)]
    dataset = datasets.Dataset.from_dict({"text": data})

    path = Path(cache_dir)
    if path.exists():
        shutil.rmtree(path)
    engine = CustomEngine(path=path, k=3)
    engine.build(vectors=vectors, corpus=dataset)

    # `search` interface
    query_vector = torch.randn(2, 8)
    output = engine.search(query_vector)
    rich.print(f"=== pipe.search() ===")
    rich.print(output)

    # `__call__` interface
    output = engine({"query": ["shs", "kkj"]}, vectors=query_vector, k=3)
    rich.print(f"=== pipe.__call__() ===")
    rich.print(output)



if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        run(Path(tmpdir) / "index")
