import math
from pathlib import Path
from typing import Dict

import datasets
import numpy as np
import pytest

from warp_pipes.search import GroupLookupSearch

n_pts = 1000
dim = 8
groups = np.linspace(0, 10, n_pts).astype(np.int64)
dataset = datasets.Dataset.from_dict({"document.gid": groups})


@pytest.mark.parametrize("cfg", [
    {'index_field': 'document', 'group_key': 'gid', 'query_field': 'query'},
])
def test_group_lookup_search_engine(cfg: Dict, tmp_path: Path):
    tmp_path = tmp_path / "test-index"
    cfg['path'] = tmp_path
    engine = GroupLookupSearch(path=tmp_path, config=cfg)
    engine.build(corpus=dataset, vectors=None)

    # process a query vector using the `search` interface
    query = np.arange(0, 11).astype(np.int64)
    scores, indices = engine.search(query, k=None)

    for i, group_idx in enumerate(query):
        scores_i, indices_i = scores[i], indices[i]
        for s_ij, idx_ij in zip(scores_i, indices_i):
            if s_ij == -math.inf:
                assert idx_ij == -1
            if idx_ij == -1:
                assert s_ij == -math.inf
            if idx_ij != -1:
                assert groups[idx_ij] == group_idx
                assert s_ij == 0
