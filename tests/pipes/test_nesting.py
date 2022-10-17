import pytest
import json
from copy import deepcopy

import numpy as np
import torch

from warp_pipes.pipes import ApplyAsFlatten, Identity, Lambda
from warp_pipes.core.condition import HasPrefix
from warp_pipes.pipes import Nested, Expand
from warp_pipes.support.datastruct import Batch


# todo: mixed types: list | np.array | Tensor
@pytest.mark.parametrize("inputs", [
    ({'document.text': [['a', 'b', 'c'], ['d', 'e', 'f']], 'question': [1, 2]}, 1),
    ({'document.text': [[['a', 'b', 'c'], ['d', 'e', 'f']]], 'question': [[1, 2]]}, 2)
])
def test_AsFlatten_identity(inputs):
    """This process a batch using the Idenity on nested fields"""
    data, level = inputs
    # with update
    pipe = ApplyAsFlatten(Identity(), update=True, input_filter=HasPrefix("document."), level=level)
    output = pipe(data)
    assert output, data

    # no update
    ref = deepcopy(data)
    ref.pop('question')
    pipe = ApplyAsFlatten(Identity(), update=False, input_filter=HasPrefix("document."), level=level)
    output = pipe(data)
    assert output, ref

@pytest.mark.parametrize("inputs", [
    ({'a': [[1, 2, 3], [1, 2, 3]], 'b': [[1, 2, 3], [1, 2, 3]]}, 1,
        {'a': [[1, 2], [1, 2]], 'b': [[1, 2], [1, 2]]}),
    ({'a': 3 * [[[1, 2, 3], [1, 2, 3]]], 'b': 3 * [[[1, 2, 3], [1, 2, 3]]]}, 2,
        {'a': 3 * [[[1, 2], [1, 2]]], 'b': 3 * [[[1, 2], [1, 2]]]}),
])
def test_Nested_drop_values(inputs):
    """Test Nested using a pipe that changes the nested batch size."""
    input, level, expected = inputs

    def drop_values(batch: Batch) -> Batch:
        """drop values >= 3"""
        def f(x):
            return x < 3

        return {k: list(filter(f, v)) for k, v in batch.items()}

    inner_pipe = Lambda(drop_values)
    pipe = Nested(inner_pipe, level=level)
    output = pipe(input)
    assert json.dumps(expected, sort_keys=True) == json.dumps(output, sort_keys=True)

@pytest.mark.parametrize("inputs", [
    ({'a': [[1, 2, 3], [1, 2, 3]], 'b': [[1, 2, 3], [1, 2, 3]]}, 1,
        {'a': [[3, 2, 1], [3, 2, 1]], 'b': [[3, 2, 1], [3, 2, 1]]}),
    ({'a': 3 * [[[1, 2, 3], [1, 2, 3]]], 'b': 3 * [[[1, 2, 3], [1, 2, 3]]]}, 2,
        {'a': 3 * [[[3, 2, 1], [3, 2, 1]]], 'b': 3 * [[[3, 2, 1], [3, 2, 1]]]}),
])
def test_Nested_sort_values(inputs):
    """Test Nested using a pipe that changes the nested batch size."""
    input, level, expected = inputs

    def sort_values(batch: Batch) -> Batch:
        """reverse sort values"""
        return {k: list(sorted(v, reverse=True)) for k, v in batch.items()}

    inner_pipe = Lambda(sort_values)
    pipe = Nested(inner_pipe, level=level)
    output = pipe(input)
    assert json.dumps(expected, sort_keys=True) == json.dumps(output, sort_keys=True)

@pytest.mark.parametrize("inputs", [
    ({'list': [1, 2, 3], 'np': np.array([1, 2, 3]), 'torch': torch.tensor([1, 2, 3])}, -1, 1, (3, 1)),
    ({'list': [1, 2, 3], 'np': np.array([1, 2, 3]), 'torch': torch.tensor([1, 2, 3])}, -1, 2, (3, 2)),
    ({'list': [1, 2, 3], 'np': np.array([1, 2, 3]), 'torch': torch.tensor([1, 2, 3])}, 0, 2, (2, 3)),
    ({'list': [1, 2, 3], 'np': np.array([1, 2, 3]), 'torch': torch.tensor([1, 2, 3])}, 0, 5,
        (5, 3)),
])
def test_Expand_shape(inputs):
    data, axis, n, expected_shape = inputs
    pipe = Expand(axis, n=n)
    output = pipe(data)
    for k in data.keys():
        assert list(expected_shape) == list(np.array(output[k]).shape)
