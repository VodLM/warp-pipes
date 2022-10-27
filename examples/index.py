from __future__ import annotations

import shutil
import string
import tempfile
from functools import partial
from pathlib import Path
from typing import Dict

import datasets
import numpy as np
import rich
import torch

from tests.utils.dummy_model import DummyModel
from warp_pipes.pipes.index import Index
from warp_pipes.support.caching import CacheConfig
from warp_pipes.support.elasticsearch import ElasticSearchInstance


def collate_fn(egs, input_key="input_ids", field: str = "query", **kwargs):
    def select_format_key(eg: Dict) -> Dict:
        return {k.replace(f"{field}.", ""): v for k, v in eg.items() if k.startswith(field)}

    egs = map(select_format_key, egs)
    inputs = [eg[input_key] for eg in egs]
    inputs = list(map(torch.tensor, inputs))
    return {input_key: torch.stack(inputs)}


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
    dataset = datasets.Dataset.from_dict(
        {"document.text": data,
         "document.input_ids": vectors}
    )

    # TODO: auto
    engines = [{
        "name": "elasticsearch",
        "config": {
            'k': 100,
            "query_filed": "query",
            "index_field": "document",
            "main_key": "text",
        }},
        {
            "name": "dense",
            "config": {
                'k': 3,
                "index_factory": "IVF1,Flat",
                "shard": False,
                "query_filed": "query",
                "index_field": "document",
            }}
    ]

    pipe = Index(
        dataset,
        engines=engines,
        model=DummyModel(input_key="input_ids", output_key="vector"),
        index_cache_config=CacheConfig(
            cache_dir=cache_dir,
            model_output_key="vector",
            collate_fn=partial(collate_fn, field="document"),
            loader_kwargs={"num_workers": 0, "batch_size": 10},
        ),
        query_cache_config=CacheConfig(
            cache_dir=cache_dir,
            model_output_key="vector",
            collate_fn=partial(collate_fn, field="query"),
            loader_kwargs={"num_workers": 0, "batch_size": 10},
        ),

    )

    # `__call__` interface
    qvectors = np.random.randn(2, dim).astype(np.float32)
    qbatch = {"query.text": data[:2],
              "input_ids": torch.from_numpy(qvectors)}
    output = pipe(qbatch)
    rich.print(f"=== pipe.__call__() ===")
    rich.print(output)

    rich.print(f"=== pipe.__call__(DATASET) ===")
    qbatch = {"query.text": ["shs", "kkj"],
              "query.input_ids": torch.from_numpy(qvectors)}
    qdataset = datasets.DatasetDict({
        "train": datasets.Dataset.from_dict({"query.text": data[:2],
                                             "query.input_ids": torch.from_numpy(
                                                 np.random.randn(2, dim).astype(np.float32))}),
        "test": datasets.Dataset.from_dict({"query.text": data[:3],
                                            "query.input_ids": torch.from_numpy(
                                                np.random.randn(3, dim).astype(np.float32))}),

    })
    output = pipe(qdataset)
    rich.print(output)
    for i in range(len(output["train"])):
        rich.print(output["train"][i])

    # TODO: implement `AsFlattenDataset`
    # nesting_level = infer_nesting(dataset, keys)
    # pipe = AsFlattenDataset(pipe, fields=["query.input_ids", "query.text"])
    # output = pipe(dataset)

if __name__ == "__main__":
    with ElasticSearchInstance():
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "index")
            run(path)
