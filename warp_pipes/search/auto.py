from os import PathLike
from typing import Any
from typing import Dict

from warp_pipes.search.base import Search
from warp_pipes.search.dense import DenseSearch
from warp_pipes.search.elasticsearch import ElasticSearch
from warp_pipes.search.group_lookup import GroupLookupSearch
from warp_pipes.search.topk import TopkSearch

Engines = {
    "dense": DenseSearch,
    "lookup": GroupLookupSearch,
    "elasticsearch": ElasticSearch,
    "topk": TopkSearch,
}


def AutoSearchEngine(
    *,
    name: str,
    path: PathLike,
    config: Dict[str, Any] = None,
    **kwargs,
) -> Search:
    # get the constructor
    EngineCls = Engines[name]
    return EngineCls(path=path, config=config, **kwargs)
