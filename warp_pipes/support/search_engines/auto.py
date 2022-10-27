from os import PathLike
from typing import Any
from typing import Dict

from warp_pipes.support.search_engines.base import SearchEngine
from warp_pipes.support.search_engines.dense import DenseSearchEngine
from warp_pipes.support.search_engines.elasticsearch import ElasticSearchEngine
from warp_pipes.support.search_engines.group_lookup import GroupLookupSearchEngine
from warp_pipes.support.search_engines.topk import TopkSearchEngine

Engines = {
    "dense": DenseSearchEngine,
    "lookup": GroupLookupSearchEngine,
    "elasticsearch": ElasticSearchEngine,
    "topk": TopkSearchEngine,
}


def AutoSearchEngine(
    *,
    name: str,
    path: PathLike,
    config: Dict[str, Any] = None,
    **kwargs,
) -> SearchEngine:
    # get the constructor
    EngineCls = Engines[name]
    return EngineCls(path=path, config=config, **kwargs)
