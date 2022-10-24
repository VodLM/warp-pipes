from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import rich

from warp_pipes.support.search_engines.dense import DenseSearchEngineConfig, DenseSearchEngine

n_pts = 1000
dim = 8
vectors = np.random.randn(n_pts, dim).astype(np.float32)


@pytest.mark.parametrize("cfg", [
    {'index_factory': "IVF1,Flat", 'shard': False},
    {'index_factory': "torch", 'shard': False},
])
def test_dense_search_engine(cfg: Dict, tmp_path: Path):
    tmp_path = tmp_path / "test-index"
    cfg['path'] = tmp_path
    cfg = DenseSearchEngineConfig(**cfg)
    engine = DenseSearchEngine(config=cfg)
    engine.build(corpus=None, vectors=vectors)
    rich.print(list(tmp_path.iterdir()))

    # process a query vector using the `search` interface
    query_vector = np.random.randn(16, dim)
    scores, indices = engine.search(query_vector, k=10)

    # compute the expected result
    m = query_vector @ vectors.T
    expected_indices = np.argsort(-m, axis=1)[:, :10]

    # check the result
    assert np.allclose(indices, expected_indices)
