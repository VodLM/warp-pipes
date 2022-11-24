from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from warp_pipes.search.dense import DenseSearchConfig, DenseSearch

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
    cfg = DenseSearchConfig(**cfg)
    engine = DenseSearch(path=tmp_path, config=cfg)
    engine.build(corpus=None, vectors=vectors)

    # process a query vector using the `search` interface
    query_vector = np.random.randn(16, dim)
    scores, indices = engine.search(query_vector, k=10)

    # compute the expected result
    m = query_vector @ vectors.T
    expected_indices = np.argsort(-m, axis=1)[:, :10]

    # check the result
    assert np.allclose(indices, expected_indices)
