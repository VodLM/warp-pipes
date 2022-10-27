from __future__ import annotations

import shutil
import string
import tempfile
from pathlib import Path

import datasets
import numpy as np
import rich
import torch

from warp_pipes.support.search_engines.dense import DenseSearchEngine


def run(cache_dir):
    path = Path(cache_dir)
    if path.exists():
        shutil.rmtree(path)

    # make dataset with random strings and random vectors
    n_pts = 1000
    dim = 8
    vectors = np.random.randn(n_pts, dim).astype(np.float32)
    data = ["".join(np.random.choice(list(string.ascii_letters), 10)
                    .tolist()) for _ in range(n_pts)]
    dataset = datasets.Dataset.from_dict({"text": data})

    engine = DenseSearchEngine(
        path=path,
        config={'k': 3, "index_factory": "IVF1,Flat", "shard": False})
    engine.build(vectors=vectors, corpus=dataset)

    # `search` interface
    query_vector = torch.randn(2, 8)
    output = engine.search(query_vector, k=3)
    rich.print(f"=== pipe.search() ===")
    rich.print(output)

    # `__call__` interface
    output = engine({"query": ["shs", "kkj"]}, vectors=query_vector, k=3)
    rich.print(f"=== pipe.__call__() ===")
    rich.print(output)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "index")
        run(path)
