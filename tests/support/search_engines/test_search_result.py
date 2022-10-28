from typing import Dict, List, Tuple
from collections import defaultdict
import pytest
import math
import torch
from warp_pipes.support.search_engines.search_result import SearchResult, sum_scores


def index_score_to_dicts(indices: torch.Tensor, scores: torch.Tensor) -> List[Dict]:
    assert torch.all(scores[indices == -1] == -math.inf)
    data = []
    for i in range(len(indices)):
        data_i = defaultdict(float)
        for idx, score in zip(indices[i], scores[i]):
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            data_i[idx] += score
        data.append(data_i)
    return data

def sum_dicts(a: Dict, b: Dict) -> Dict:
    c = defaultdict(float)
    for k,v in list(a.items()) + list(b.items()):
        c[k] += v
    return c

def sum_dicts_list(list_a: List[Dict], list_b: List[Dict]) -> Dict:
    output = []
    for a,b in zip(list_a, list_b):
        output.append(sum_dicts(a, b))
    return output


@pytest.mark.parametrize("a, b", [
    (
        (
            torch.tensor([[0, 1, 2], [10, 12, -1]]),
            torch.tensor([[100.1, 101.2, 102.3], [10.1, 10.2, -math.inf]])
        ),
        (
            torch.tensor([[2, 3, -1], [12, 13, -1]]),
            torch.tensor([[102.1, 103.2, -math.inf], [12.1, 13.2, -math.inf]])
        )
    ),
    (
        (
            torch.tensor([[0, 1, 2, 4, 5], [10, 12, -1, 56, 57]]),
            torch.tensor([[100.1, 101.2, 102.3, 59.3, 23.4], [10.1, 10.2, -math.inf, -math.inf, -234.0]])
        ),
        (
            torch.tensor([[2, 3, -1], [12, 13, -1]]),
            torch.tensor([[102.1, 103.2, -math.inf], [12.1, 13.2, -math.inf]])
        )
    ),
])
def test_sum_scores(a: Tuple[torch.Tensor, torch.Tensor], b: Tuple[torch.Tensor, torch.Tensor]):
    """Test summing scores referenced by indices (See: `SearchResults`). The test consists of
    testing the summation using a basic python dictionaries."""

    # compute the expected output using the naive implementation
    a_dict_list = index_score_to_dicts(*a)
    b_dict_list = index_score_to_dicts(*b)
    excepted = sum_dicts_list(a_dict_list, b_dict_list)

    # compute using the function under test
    indices, scores = sum_scores(a, b)
    output = index_score_to_dicts(indices, scores)

    assert len(excepted) == len(output)
    for i in range(len(excepted)):
        for key in excepted[i].keys() |  output[i].keys():
            if key == -1:
                assert excepted[i][key] == -math.inf
            else:
                assert excepted[i][key] == output[i][key]
