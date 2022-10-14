import re
from typing import Callable
from typing import Dict
from typing import Optional

from warp_pipes.support.datastruct import Batch


def always_true(*args, **kwargs):
    return True


def get_batch_eg(batch: Batch, idx: int, filter_op: Optional[Callable] = None) -> Dict:
    """Extract example `idx` from a batch, potentially filter the keys"""
    filter_op = filter_op or always_true
    return {k: v[idx] for k, v in batch.items() if filter_op(k)}


def camel_to_snake(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
