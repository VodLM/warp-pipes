from copy import copy
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import T
from typing import Tuple
from typing import Union

from omegaconf import DictConfig


def copy_with_override(data: Dict, key: Any, value: Any) -> Dict:
    data = copy(data)
    data[key] = value
    return data


def apply_to_json_struct(
    data: Union[List, Dict], fn: Callable, strict=False, **kwargs
) -> Union[List, Dict]:
    """
    Apply a function to a json-like structure
    Parameters
    ----------
    data
        json-like structure
    fn
        function to apply
    kwargs
        keyword arguments to pass to fn

    Returns
    -------
    json-like structure
    """
    if isinstance(data, (dict, DictConfig)):
        try:
            output = {
                key: apply_to_json_struct(
                    value, fn, **copy_with_override(kwargs, "key", key)
                )
                for key, value in data.items()
            }
        except Exception:
            if strict:
                raise
            output = {
                key: apply_to_json_struct(value, fn, **kwargs)
                for key, value in data.items()
            }

        return output
    elif isinstance(data, list):
        return [apply_to_json_struct(value, fn, **kwargs) for value in data]
    else:
        return fn(data, **kwargs)


def flatten_json_struct(data: Union[List, Dict]) -> Iterable[Any]:
    """
    Flatten a json-like structure
    Parameters
    ----------
    data
        json-like structure
    Yields
    -------
    Any
        Leaves of json-like structure
    """
    if isinstance(data, dict):
        for key, x in data.items():
            for leaf in flatten_json_struct(x):
                yield leaf
    elif isinstance(data, list):
        for x in data:
            for leaf in flatten_json_struct(x):
                yield leaf
    else:
        yield data


def get_named_attributes(
    data: Union[List, Dict],
    key_filter: Optional[Callable[[str], bool]] = None,
    current_key: Optional[str] = None,
) -> Iterable[Tuple[str, Any]]:
    """
    Get the flatten named attributes from a json-like structure.

    NB: Only attributes registered as a dictionary (with a key) are returned.

    Parameters
    ----------
    data
    key_filter
    current_key

    Returns
    -------

    """
    if isinstance(data, dict):
        for key, x in data.items():
            for leaf in get_named_attributes(x, current_key=key, key_filter=key_filter):
                yield leaf
    elif isinstance(data, list):
        for x in data:
            for leaf in get_named_attributes(x, key_filter=key_filter):
                yield leaf
    elif current_key is not None:
        if key_filter is None or key_filter(current_key):
            yield (current_key, data)
    else:
        pass


def reduce_json_struct(
    data: Union[List, Dict],
    reduce_op: Callable[[Iterable[T]], T],
    key_filter: Optional[Callable[[T], bool]] = None,
) -> T:
    """
    Reduce a json-like structure
    Parameters
    ----------
    data
        json-like structure
    reduce_op
        reduce operation
    Returns
    -------
    reduced json-like structure
    """
    if key_filter is not None:
        named_leaves = get_named_attributes(data, key_filter=key_filter)
        leaves = (v for k, v in named_leaves)
    else:
        leaves = flatten_json_struct(data)
    return reduce_op(leaves)
