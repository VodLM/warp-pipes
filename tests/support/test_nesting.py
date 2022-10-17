import pytest
import json


import numpy as np
import torch

from warp_pipes.support.nesting import flatten_nested, nested_list, expand_and_repeat


@pytest.mark.parametrize(
    "inputs",
    [
        ([[1, 2, 3], [4, 5, 6]], 1, [1, 2, 3, 4, 5, 6]),
        ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 1, [[1, 2], [3, 4], [5, 6], [7, 8]]),
        ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2, [1, 2, 3, 4, 5, 6, 7, 8]),
    ],
)
def test_flatten_nested(inputs):
    values, level, expected = inputs
    output = list(flatten_nested(values, level=level))
    assert json.dumps(output) == json.dumps(expected)


@pytest.mark.parametrize(
    "inputs",
    [
        (list(range(64)), (-1, 8)),
        (list(range(64)), (8, -1)),
        (list(range(64)), (8, -1, 2, 2)),
        (list(range(64)), (2, 2, 2, 2, -1)),
    ],
)
def test_nested_list(inputs):
    values, shape = inputs
    expected = np.array(values).reshape(shape)
    output = np.array(nested_list(values, shape=shape))
    assert (expected == output).all()


@pytest.mark.parametrize(
    "inputs",
    [
        ([1, 2, 3], -1, 4, [4 * [1], 4 * [2], 4 * [3]], None),
        ([1, 2, 3], -1, 4, [4 * [1], 4 * [2], 4 * [3]], np.array),
        ([1, 2, 3], -1, 4, [4 * [1], 4 * [2], 4 * [3]], torch.tensor),
        ([[1, 2, 3]], -1, 4, [[4 * [1], 4 * [2], 4 * [3]]], None),
        ([[1, 2, 3]], -1, 4, [[4 * [1], 4 * [2], 4 * [3]]], np.array),
        ([[1, 2, 3]], -1, 4, [[4 * [1], 4 * [2], 4 * [3]]], torch.tensor),
        ([1, 2, 3], 0, 4, 4 * [[1, 2, 3]], None),
        ([1, 2, 3], 0, 4, 4 * [[1, 2, 3]], np.array),
        ([1, 2, 3], 0, 4, 4 * [[1, 2, 3]], torch.tensor),
        ([[1, 2, 3]], 0, 4, 4 * [[[1, 2, 3]]], None),
        ([[1, 2, 3]], 0, 4, 4 * [[[1, 2, 3]]], np.array),
        ([[1, 2, 3]], 0, 4, 4 * [[[1, 2, 3]]], torch.tensor),
        ([1, 2, 3], 1, 4, [4 * [1], 4 * [2], 4 * [3]], None),
        ([1, 2, 3], 1, 4, [4 * [1], 4 * [2], 4 * [3]], np.array),
        ([1, 2, 3], 1, 4, [4 * [1], 4 * [2], 4 * [3]], torch.tensor),
        ([[1, 2, 3]], 1, 4, [4 * [[1, 2, 3]]], None),
        ([[1, 2, 3]], 1, 4, [4 * [[1, 2, 3]]], np.array),
        ([[1, 2, 3]], 1, 4, [4 * [[1, 2, 3]]], torch.tensor),
    ],
)
def test_expand_and_repeat(inputs):
    data, axis, n, expected, op = inputs
    if op is not None:
        data = op(data)
        expected = op(expected)

    # process
    output = expand_and_repeat(data, axis=axis, n=n)

    # cast
    output = np.array(output)
    expected = np.array(expected)

    # test
    assert (expected == output).all()
    assert expected.shape == output.shape
