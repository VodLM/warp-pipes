from __future__ import annotations
from typing import Dict, Set, Tuple

import pickle
from typing import List
from typing import Optional, Any

import pytest
import rich
import dill
import datasets

from warp_pipes.core.condition import In
from warp_pipes.core.pipe import Pipe
from warp_pipes.support.datastruct import Batch, Eg
from warp_pipes.support.shapes import infer_batch_size


class DummyPipe(Pipe):
    """A simplistic pipe that returns a dummy value."""

    output_key = "dummy"

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        batch_size = infer_batch_size(batch)
        iid = "".join(sorted(str(key) for key in batch.keys()))
        data = [f"{iid}-{1+i}" for i in range(batch_size)]
        return {DummyPipe.output_key: data}

    def _call_egs(self, egs: List[Eg], **kwargs) -> Batch:
        iid = "".join(sorted(str(key) for key in egs[0].keys()))
        data = [f"{iid}-{1+i}" for i in range(len(egs))]
        return {DummyPipe.output_key: data}

    @classmethod
    def instantiate_test(cls, **kwargs):
        """Instantiate a test pipe."""
        return cls()


@pytest.mark.parametrize(
    "inputs",
    [
        (
            {"a": [1, 2, 3], "b": [4, 5, 6]},
            {DummyPipe.output_key: ["ab-1", "ab-2", "ab-3"]},
            {"update": False, "input_filter": None},
        ),
        (
            {"a": [1, 2, 3], "b": [4, 5, 6]},
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                DummyPipe.output_key: ["ab-1", "ab-2", "ab-3"],
            },
            {"update": True, "input_filter": None},
        ),
        (
            {"a": [1, 2, 3], "b": [4, 5, 6]},
            {DummyPipe.output_key: ["a-1", "a-2", "a-3"]},
            {"update": False, "input_filter": In(["a"])},
        ),
        (
            {"a": [1, 2, 3], "b": [4, 5, 6]},
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                DummyPipe.output_key: ["a-1", "a-2", "a-3"],
            },
            {"update": True, "input_filter": In(["a"])},
        ),
    ],
)
def test_process_batch(inputs: Tuple[Batch, Batch, Dict[str, Any]]):
    """Process an input batch and check the output for different Pipe configurations."""
    input, expected_output, kwargs = inputs
    pipe = DummyPipe(**kwargs)
    output = pipe(input)
    assert set(output.keys()) == set(expected_output.keys())
    for key in expected_output.keys():
        assert output[key] == expected_output[key]


@pytest.mark.parametrize(
    "inputs",
    [
        (
            [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}],
            {DummyPipe.output_key: ["ab-1", "ab-2", "ab-3"]},
            {"update": False, "input_filter": None},
        ),
    ],
)
def test_process_egs(inputs: Tuple[List[Eg], Batch, Dict[str, Any]]):
    """Process an input list of examples and check the output for different Pipe configurations."""
    input, expected_output, kwargs = inputs
    pipe = DummyPipe(**kwargs)
    output = pipe(input)
    assert set(output.keys()) == set(expected_output.keys())
    for key in expected_output.keys():
        assert output[key] == expected_output[key]


@pytest.mark.parametrize(
    "inputs",
    [
        (
            datasets.Dataset.from_dict({"a": [1, 2, 3, 4], "b": [5, 6, 7,8]}),
            {DummyPipe.output_key: ["ab-1", "ab-2", "ab-1", "ab-2"]},
            {"update": False, "input_filter": None},
        ),
    ],
)
def test_process_dataset(inputs: Tuple[datasets.Dataset, Batch, Dict[str, Any]]):
    """Test processing a `datasets.Dataset` using n=2 processes (NB: affects the expected output)"""
    input, expected_output, kwargs = inputs
    pipe = DummyPipe(**kwargs)
    output = pipe(input, num_proc=2, batch_size=2)
    assert set(output.column_names) == (
        set(output.column_names) | set(expected_output.keys())
    )
    for key in expected_output.keys():
        assert output[key] == expected_output[key]


@pytest.mark.parametrize(
    "inputs",
    [
        (
            datasets.DatasetDict(
                {"train": datasets.Dataset.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})}
            ),
            None,
            {"update": False, "input_filter": None},
        ),
    ],
)
def test_process_dataset_dict(
    inputs: Tuple[datasets.DatasetDict, None, Dict[str, Any]]
):
    input, _, kwargs = inputs
    pipe = DummyPipe(**kwargs)
    output = pipe(input, num_proc=2, batch_size=2)
    assert output.keys() == input.keys()


def test_Pipe_num_proc():
    """Test `Pipe.num_proc`."""
    # level one
    pipe = Pipe()
    pipe._max_num_proc = 3
    assert pipe.max_num_proc == 3

    # level two
    sub_pipe = Pipe()
    sub_pipe._max_num_proc = 2
    pipe.sub_pipe = sub_pipe
    assert pipe.max_num_proc == 2

    # level three
    sub_sub_pipe = Pipe()
    sub_sub_pipe._max_num_proc = 1
    sub_pipe.sub_pipe = sub_sub_pipe
    assert pipe.max_num_proc == 1
