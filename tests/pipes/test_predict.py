import tempfile
import pytest
import rich
import numpy as np
import dill
import torch
import datasets
from tests.utils.dummy_model import DummyModel
from warp_pipes.pipes import PredictWithoutCache, PredictWithCache
from warp_pipes.pipes.basics import ApplyToAll
from warp_pipes.pipes.pipelines import Sequential
from warp_pipes.support.pretty import pprint_batch

base_cfg = {
    "input_key": "data",
    "output_key": "vector",
    "call_kwargs": {},
}
data = np.random.randn(100, 8).astype(np.float32)
batch = {base_cfg["input_key"]: data}
dataset = datasets.Dataset.from_dict(batch)
model = DummyModel(
    data.shape[1], input_key=base_cfg["input_key"], output_key=base_cfg["output_key"]
)
with torch.inference_mode():
    preds = model({base_cfg["input_key"]: torch.from_numpy(data)})
    if base_cfg["output_key"] is not None:
        preds = preds[base_cfg["output_key"]]
    preds = preds.detach().numpy()
    batch_preds = {base_cfg["output_key"]: preds}
    dataset_preds = datasets.Dataset.from_dict(batch_preds)
    dataset_preds = datasets.concatenate_datasets([dataset, dataset_preds], axis=1)


def collate_fn(egs, input_key="data", **kwargs):
    inputs = [eg[input_key] for eg in egs]
    inputs = list(map(torch.tensor, inputs))
    return {input_key: torch.stack(inputs)}


@torch.inference_mode()
@pytest.mark.parametrize(
    "cfg",
    [
        # {
        #     **base_cfg,
        #     "input": batch,
        #     "output": batch_preds,
        #     "call_kwargs": {"idx": list(range(len(data)))},
        # },
        # {
        #     **base_cfg,
        #     "input": dataset,
        #     "output": dataset_preds,
        #     "call_kwargs": {"batch_size": 10, "num_proc": 1},
        # },
        {
            **base_cfg,
            "input": dataset,
            "output": dataset_preds,
            "call_kwargs": {"batch_size": 10, "num_proc": 2},
        },
        # {
        #     **base_cfg,
        #     "input": datasets.DatasetDict({"train": dataset}),
        #     "output": datasets.DatasetDict({"train": dataset_preds}),
        #     "call_kwargs": {"batch_size": 10, "num_proc": 1},
        # },
    ],
)
@pytest.mark.parametrize(
    "model_info",
    [
        # (PredictWithoutCache, {}),
        (
            PredictWithCache,
            {
                "caching_kwargs": {
                    "model_output_key": base_cfg["output_key"],
                    "collate_fn": collate_fn,
                    "loader_kwargs": {"batch_size": 10, "num_workers": 0},
                },
            },
        ),
    ],
)
def test_predict_pipes(cfg, model_info):
    """Test PredictWithoutCache."""
    Cls, kwargs = model_info
    with tempfile.TemporaryDirectory() as cache_dir:

        if Cls == PredictWithCache:
            kwargs["caching_kwargs"]["cache_dir"] = cache_dir

        predict_pipe = Cls(model, **kwargs)

        if isinstance(predict_pipe, PredictWithoutCache):
            predict_pipe = Sequential(ApplyToAll(torch.tensor), predict_pipe)

        elif isinstance(predict_pipe, PredictWithCache):
            cache_fingerprint = predict_pipe.cache(dataset)
            cfg["call_kwargs"]["cache_fingerprint"] = cache_fingerprint


        if not dill.pickles(predict_pipe):
            raise ValueError("predict_pipe does not pickle")

        # process input with pipe
        output = predict_pipe(cfg["input"], **cfg["call_kwargs"])

        # check output
        if isinstance(cfg["output"], datasets.DatasetDict):
            assert cfg["output"].keys() == output.keys()
            output = output["train"]
            cfg["output"] = cfg["output"]["train"]

        if isinstance(cfg["output"], dict):
            assert cfg["output"].keys() == output.keys()
            for k in cfg["output"].keys():
                assert np.allclose(cfg["output"][k], output[k], rtol=1.e-2, atol=1.e-2)

        elif isinstance(cfg["output"], datasets.Dataset):
            assert cfg["output"].column_names == output.column_names
            for k in cfg["output"].column_names:
                assert cfg["output"][k] == output[k]
            #     pprint_batch({ "input": cfg["output"][k], "output": output[k] }, header=k)
            #     if not cfg["output"][k] == output[k]:
            #         rich.print(f"Failed: {k}")
            #         for i in range(100):
            #             rich.print(f"{i} : {cfg['output'][k][i]} != {output[k][i]}")

        else:
            raise TypeError(f"Unexpected type {type(cfg['output'])}")
