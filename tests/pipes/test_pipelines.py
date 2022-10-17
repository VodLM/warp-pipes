
import pytest
from warp_pipes.pipes import Gate, Identity
from warp_pipes.pipes.basics import Lambda


@pytest.mark.parametrize("inputs", [
    ({'a': [1, 2, 3], 'b': [4, 5, 6]}, True, Identity(), None, {'a': [1, 2, 3], 'b': [4, 5, 6]}),
    ({'a': [1, 2, 3], 'b': [4, 5, 6]}, False, Identity(), None, {}),
    ({'a': [1, 2, 3], 'b': [4, 5, 6]}, False, Identity(), Lambda(lambda x: {"z": [1,2,3]}), {"z": [1,2,3]}),
    ({'a': [1, 2, 3], 'b': [4, 5, 6]}, lambda x: True, Identity(), None, {'a': [1, 2, 3], 'b': [4, 5, 6]}),
    ({'a': [1, 2, 3], 'b': [4, 5, 6]}, lambda x: False, Identity(), None, {}),
])
def test_Gate_call(inputs):
    batch, cond, pipe, alt, expected = inputs
    # build pipe and process batch
    gated_pipe = Gate(cond, pipe=pipe, alt=alt)
    output = gated_pipe(batch)

    # compare output to expected output
    assert set(expected.keys()) == set(output.keys())
