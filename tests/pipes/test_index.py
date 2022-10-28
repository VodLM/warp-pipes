import tempfile
from copy import copy
from functools import partial
from typing import Dict

import datasets
import numpy as np
import pytest
import rich
import torch

from tests.utils.dummy_model import DummyModel
from warp_pipes.pipes import Index
from warp_pipes.support.shapes import infer_shape

base_cfg = {
    "input_keys": ["document.data", "query.data"],
    "input_query_key": "query.data",
    "output_key": "vector",
}

base_loader_kwargs = {
    "batch_size": 10,
    "num_workers": 0,
}


def gen_dataset(field: str, n_pts: int):
    data = np.random.randn(n_pts, 8).astype(np.float32)
    batch = {f"{field}.data": data}
    corpus = datasets.Dataset.from_dict(batch)
    return corpus


def gen_model(cfg: Dict):
    model = DummyModel(
        8,
        input_keys=base_cfg["input_keys"],
        output_key=base_cfg["output_key"],
    )
    return model


def collate_fn(egs, input_key="data", **kwargs):
    inputs = [eg[input_key] for eg in egs]
    inputs = list(map(torch.tensor, inputs))
    return {input_key: torch.stack(inputs)}


@torch.inference_mode()
@pytest.mark.parametrize(
    "cfg",
    [
        {**base_cfg,
         'engines': [{'name': 'dense',
                      'config': {
                          'k': 50,
                          'index_factory': 'torch',
                          'index_field': 'document',
                          'query_field': 'query',
                          'score_key': "score",
                          'index_key': 'idx',
                          'verbose': 'false',
                      }},
                     {'name': 'topk',
                      'config': {
                          'k': 10,
                          'index_factory': 'torch',
                          'index_field': 'document',
                          'query_field': 'query',
                          'score_key': "score",
                          'index_key': 'idx',
                          'verbose': 'false',
                          'merge_with_previous_results': 'false',
                      }}
                     ],
         'index_cache_config': {
             'collate_fn': partial(collate_fn, input_key="document.data"),
             'model_output_key': base_cfg["output_key"],
             'loader_kwargs': base_loader_kwargs,
         },
         'query_cache_config': {
             'collate_fn': partial(collate_fn, input_key="query.data"),
             'model_output_key': base_cfg["output_key"],
             'loader_kwargs': base_loader_kwargs,
         }
         }
    ],
)
def test_predict_pipes(cfg):
    """Test PredictWithoutCache."""
    cfg = copy(cfg)

    # gen dataset & model
    corpus = gen_dataset("document", 100)
    queries = gen_dataset("query", 50)
    model = gen_model(cfg)

    # init the pipe
    with tempfile.TemporaryDirectory() as cache_dir:
        pipe = Index(
            corpus=corpus,
            cache_dir=cache_dir,
            model=model,
            engines=cfg['engines'],
            index_cache_config=cfg['index_cache_config'],
            query_cache_config=cfg['query_cache_config'],
        )

        # run through the pipe
        output = pipe(queries)

        # validate the output
        expected_output_names = set(queries.column_names) | {"document.score", "document.idx"}
        assert set(output.column_names) == expected_output_names
        assert infer_shape(output["document.score"]) == [len(queries), 10]
