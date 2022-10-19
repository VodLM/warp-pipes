import pytest
import rich
import numpy as np
import torch
import datasets
from tests.utils.dummy_model import DummyModel
from warp_pipes.pipes import PredictWithoutCache
from warp_pipes.pipes.basics import ApplyToAll
from warp_pipes.pipes.pipelines import Sequential

base_cfg = {
    "input_key": "data",
    "output_key": "vector",
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
    batch_preds =  {base_cfg["output_key"]: preds}
    dataset_preds = datasets.Dataset.from_dict(batch_preds)
    dataset_preds = datasets.concatenate_datasets([dataset, dataset_preds], axis=1)


@torch.inference_mode()
@pytest.mark.parametrize(
    "cfg",
    [
        {**base_cfg, "input": batch, "output": batch_preds},
        {
            **base_cfg,
            "input": dataset,
            "output": dataset_preds,
            "kwargs": {"batch_size": 10, "num_proc": 1},
        },
        {
            **base_cfg,
            "input": dataset,
            "output": dataset_preds,
            "kwargs": {"batch_size": 10, "num_proc": 2},
        },
        {
            **base_cfg,
            "input": datasets.DatasetDict({"train": dataset}),
            "output": datasets.DatasetDict({"train": dataset_preds}),
        },
    ],
)
def test_PredictWithoutCache(cfg):
    """Test PredictWithoutCache."""
    predict_pipe = PredictWithoutCache(model)
    pipe = Sequential(ApplyToAll(torch.tensor), predict_pipe)
    output = pipe(cfg["input"])

    if isinstance(cfg["output"], datasets.DatasetDict):
        assert cfg["output"].keys() == output.keys()
        output = output["train"]
        cfg["output"] = cfg["output"]["train"]

    if isinstance(cfg["output"], dict):
        assert cfg["output"].keys() == output.keys()
        for k in cfg["output"].keys():
            assert np.allclose(cfg["output"][k], output[k])

    elif isinstance(cfg["output"], datasets.Dataset):
        assert cfg["output"].column_names == output.column_names
        for k in cfg["output"].column_names:
            assert np.allclose(cfg["output"][k], output[k])

    else:
        raise TypeError(f"Unexpected type {type(cfg['output'])}")
