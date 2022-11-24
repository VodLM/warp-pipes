import numpy as np
import pytest

from warp_pipes.support.shapes import infer_missing_dims, infer_shape_nested_list, \
    infer_batch_shape, infer_nesting_level


@pytest.mark.parametrize("inputs", [
    (64, [-1, 8], [8, 8]),
    (64, [8, -1], [8, 8]),
    (128, [8, -1, 2], [8, 8, 2]),
    (128, [8, 2, -1], [8, 2, 8]),

])
def test_infer_missing_dims(inputs):
    n_elements, shape, expected_shape = inputs
    new_shape = infer_missing_dims(n_elements=n_elements, shape=shape)
    assert np.prod(new_shape) == n_elements
    assert new_shape == expected_shape

@pytest.mark.parametrize("values", [
    ([1, 2, 3, 4],),
    ([[1, 2, 3], [4, 5, 6]],),
    ([[[1, 2], [3, 4]], [[4, 5], [6, 7]]],),
    ([[1, 2, 3], [4, 5]],),
    ([[[1], 2], [4, 5]],),
])
def test_infer_shape_nested_list(values):
    shape = infer_shape_nested_list(values)
    assert shape == list(np.array(values).shape)

@pytest.mark.parametrize("inputs", [
    ({'a': [1, 2], 'b': [[1, 2, 3], [1, 2]]}, [2]),
    ({'a': [[[1, 1], [1, 1]], [[2, 2], [2, 2]]],
        'b': [[[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [[[2, 2], [2, 2]], [[2, 2], [2, 2]]]]},
        [2, 2, 2]),
    ({'document.text': [['jkhnds', 'aff', 'asdsd']],
        'document.input_ids': [[[1, 2], [1, 2, 3], [1, 2, 3, 4]]],
        'question.text': ['bajk'],
        'question.input_ids': [[1, 2, 3]]
        }, [1])
]
)
def test_infer_batch_shape(inputs):
    batch, expected_shape =  inputs
    assert infer_batch_shape(batch) == expected_shape


@pytest.mark.parametrize("batch,nesting_level", [
    ({"a": [1, 2, 3]}, 0),
    ({"b": [["abc", "def", "gh"], ["abc", "def", "gh"]]}, 1),
    ({"a": [1, 2,], "b": [["abc", "def", "gh"], ["abc", "def", "gh"]]}, 0),
])
def test_infer_batch_shape(batch, nesting_level):
    assert infer_nesting_level(batch) == nesting_level
