import math
import os
import string
import subprocess
import time
from collections import defaultdict
from copy import copy
from typing import AsyncIterator, Coroutine, Iterable, AsyncIterable
from typing import Dict
from typing import List
from typing import Optional

import rich
import torch
from elasticsearch import AsyncElasticsearch
from elasticsearch import RequestError
from elasticsearch import helpers as es_helpers
from loguru import logger
from pydantic import BaseModel

from warp_pipes.support.datastruct import Batch
from warp_pipes.support.datastruct import Eg
from warp_pipes.support.functional import iter_batch_egs
from warp_pipes.support.shapes import infer_batch_size

VERBOSE = False


class EsSearchFnConfig(BaseModel):
    """
    Args:
        query_key (:obj:`str`): name of the elasticsearch field to query and name of the key in the
            batch
        auxiliary_key (:obj:`str`): name of the batch field used to query the main elasticsearch
            field (`query_key`)
        filter_key (:obj:`str`): name of the field used for filtering
        auxiliary_weight (:obj:`float`): value of the weight (beta)
        scale_auxiliary_weight_by_lengths (:obj:`bool`): scale the auxiliary weight by the ratio
            of the lengths `query`  / `aux_query`
        k (:obj:`int`): number of results
        index_name (:obj:`str`): name of the elasticsearch index
        output_keys (:obj:`List[str]`): list of keys to return from ES (limit I/O)
    """
    k: int
    index_name: str
    query_key: str
    auxiliary_key: Optional[str] = None
    filter_key: Optional[str] = None
    auxiliary_weight: float = 0,
    scale_auxiliary_weight_by_lengths: bool = False
    output_keys: Optional[List[str]] = None,


async def es_search(
        batch: Batch,
        *,
        es_instance: AsyncElasticsearch,
        es_search_fn_config: EsSearchFnConfig,
        request_timeout: int = 600,
        chunk_size: int = 100,
) -> Batch:
    """
    Search an Elasticsearch index based on the field `query_key` and, secondarily, `auxiliary_key`
    and `filter_key`.

    Without filtering the ranking function is
    ```python

    r(batch, doc) = BM25(batch[query_key], doc[query_key]) + beta *
        BM25(batch[auxiliary_query_key], doc)[query_key]
    ```
    where
    ```python
    # when `scale_auxiliary_weight_by_lengths` is False
    beta = auxiliary_query_key
    # when `scale_auxiliary_weight_by_lengths` is True
    beta = _get_scaled_auxiliary_weight(auxiliary_weight, batch[query_key],
    batch[auxiliary_query_key])
    ```
    With filtering, only documents matching `batch[filter_key]` on the field `filter_key` are
    returned.

    Args:
        chunk_size (:obj:`int`): number of queries to send to ES at once
        batch (:obj:`Batch`): input query
        es_instance (:obj:`AsyncElasticsearch`): AsyncElasticsearch instance
        es_search_fn_config (:obj:`EsSearchFnConfig`): configuration of the search function
        request_timeout (:obj:`int`): timeout value
    """
    if es_search_fn_config.auxiliary_weight > 0 and es_search_fn_config.auxiliary_key not in batch:
        raise ValueError(
            f"key `{es_search_fn_config.auxiliary_key}` must be provided if auxiliary_weight > 0, "
            f"Found keys {batch.keys()}"
        )

    async def make_chunks_of_es_search_requests(batch: Batch, chunk_size: int) -> AsyncIterable[List[Coroutine]]:
        """Iterate over the batch and make groups of requests"""
        chunk = []
        idx = 0
        count = 0
        batch_size = infer_batch_size(batch)
        for eg in iter_batch_egs(batch):
            r = _make_query_request(eg, es_search_fn_config)
            count += 1
            chunk.append(r)
            if count == chunk_size:
                idx += 1
                if VERBOSE:
                    rich.print(f'>> yield: {idx} ({idx * chunk_size} / {batch_size})')
                yield chunk
                chunk = []
                count = 0

        if len(chunk):
            idx += 1
            if VERBOSE:
                rich.print(f'>> yield: {idx} (last)')
            yield chunk

    async def search_and_parse(futures_requests: List[Coroutine]):
        """Search and parse the results"""
        def join_requests(reqs: List[List[Dict]]) -> List[Dict]:
            """Join the requests"""
            return [r for req in reqs for r in req]

        requests = join_requests([await f for f in futures_requests])
        return await es_instance.msearch(
            body=requests, request_timeout=request_timeout
        )

    async def format_es_response(response: Dict) -> Batch:
        batch_of_results = defaultdict(list)
        for item_response in response["responses"]:
            if "hits" not in item_response:
                raise ValueError(f"ES did not return any hits. Response: {item_response}")

            result_i = defaultdict(list)
            for hit in item_response["hits"]["hits"]:
                hit_data = {"scores": hit["_score"], **hit["_source"]}
                for k, v in hit_data.items():
                    result_i[k].append(v)

            for k, v in result_i.items():
                batch_of_results[k].append(v)
        return batch_of_results

    async def search_batch_with_async_chunks(batch: Batch, chunk_size: int) -> Batch:
        """Search all the requests in the batch"""
        idx = 0
        futures = []
        async for requests in make_chunks_of_es_search_requests(batch, chunk_size):
            idx += 1
            if VERBOSE:
                rich.print(f'>> search:send {idx} ({chunk_size})')
            r = search_and_parse(requests)
            futures.append(r)

        all_results = defaultdict(list)
        for idx, f in enumerate(futures):
            r = await f
            r = await format_es_response(r)
            if VERBOSE:
                rich.print(f">> Received {idx} ({chunk_size})")
            for key, value in r.items():
                all_results[key].extend(value)
        return all_results

    return await search_batch_with_async_chunks(batch, chunk_size)


async def _make_query_request(eg, config: EsSearchFnConfig) -> List[Dict]:
    use_aux_queries = config.auxiliary_weight > 0
    should_query_parts = []
    filter_query_parts = []
    # make the main query
    query = eg[config.query_key]
    should_query_parts.append(
        {
            "match": {
                config.query_key: {
                    "query": query,
                    "operator": "or",
                }
            }
        },
    )
    # make the auxiliary query
    if use_aux_queries:
        aux_query = eg[config.auxiliary_key]
        if config.scale_auxiliary_weight_by_lengths:
            aux_weight_i = _get_scaled_auxiliary_weight(
                config.auxiliary_weight, query, aux_query
            )
        else:
            aux_weight_i = config.auxiliary_weight

        should_query_parts.append(
            {
                "match": {
                    config.query_key: {
                        "query": aux_query,
                        "operator": "or",
                        "boost": aux_weight_i,
                    }
                }
            },
        )
    # make the filter query
    if config.filter_key is not None:
        filter_query = eg[config.filter_key]
        if isinstance(filter_query, torch.Tensor):
            filter_query = filter_query.item()
        filter_query_parts.append({"term": {config.filter_key: filter_query}})

    # output keys (only return the keys that are needed to reduce the size of the response)
    if config.output_keys is not None:
        output_query_part = {"_source": config.output_keys}
    else:
        output_query_part = {}

    # make the final request
    r = {
        "query": {
            "bool": {"should": should_query_parts, "filter": filter_query_parts},
        },
        **output_query_part,
        "from": 0,
        "size": config.k,
    }
    return [{"index": config.index_name}, r]


async def es_create_index(
        index_name: str, *, es_instance: AsyncElasticsearch, body: Optional[Dict] = None
) -> bool:
    try:
        response = await es_instance.indices.create(index=index_name, body=body)
        logger.info(response)
        newly_created = True

    except RequestError as err:
        if err.error == "resource_already_exists_exception":
            newly_created = False
        else:
            raise err

    return newly_created


async def es_remove_index(index_name: str, *, es_instance: AsyncElasticsearch):
    logger.info(f"Removing index {index_name}")
    deleted_ok = await es_instance.indices.delete(index=index_name)
    logger.info(f"Index {index_name} removed: {deleted_ok}")
    return deleted_ok


async def es_ingest(
        egs: AsyncIterator[Eg],
        *,
        es_instance: AsyncElasticsearch,
        index_name: str,
):
    async def gen_actions(egs, index_name):
        async for eg in egs:
            yield {"_index": index_name, "_source": eg}

    async for ok, result in es_helpers.async_streaming_bulk(
            es_instance,
            gen_actions(egs, index_name),
            max_retries=10,
    ):
        action, result = result.popitem()
        if not ok:
            raise ValueError(f"Failed to {action}: {result}")


def ping_es(host=None, **kwargs):
    if host is None:
        hosts = None
    else:
        hosts = [host]
    return AsyncElasticsearch(hosts=hosts).ping(**kwargs)


class ElasticSearchInstance(object):
    """Instantiate ElasticSearch as a subprocess."""

    TIMEOUT = 3600

    def __init__(self, disable: bool = False, **kwargs):
        self.disable = disable
        self.kwargs = copy(kwargs)

    def __enter__(self):
        # make a database connection and return it
        if ping_es():
            logger.info("Elasticsearch is already running")
            return

        if not self.disable:
            env = copy(os.environ)
            cmd = "elasticsearch"
            logger.info(
                f"Spawning ElasticSearch: {cmd}, "
                f"ES_JAVA_OPTS={env.get('ES_JAVA_OPTS', '<none>')}"
            )
            self.es_proc = subprocess.Popen([cmd], env=env, **self.kwargs)
            t0 = time.time()
            while not ping_es():
                time.sleep(0.5)
                if time.time() - t0 > self.TIMEOUT:
                    raise TimeoutError("Couldn't ping the ES instance.")

            logger.info(
                f"Elasticsearch is up and running "
                f"(init time={time.time() - t0:.1f}s)"
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the db connection gets closed
        if hasattr(self, "es_proc"):
            logger.info("Terminating elasticsearch process")
            self.es_proc.terminate()


def _tokenize(text: str) -> List[str]:
    """Filter Punctuation and split text into tokens"""
    tokens = text.split(" ")
    tokens = [
        token.translate(str.maketrans("", "", string.punctuation)) for token in tokens
    ]
    tokens = [token for token in tokens if token.strip() != ""]
    return tokens


def _get_scaled_auxiliary_weight(
        auxiliary_weight: float, query: str, aux_query: str
) -> float:
    """Scale the auxiliary weight with the length of the query and the auxiliary query"""
    query_length = len(_tokenize(query))
    aux_query_length = len(_tokenize(aux_query))
    if auxiliary_weight > 0 and aux_query_length > 0:
        r = query_length / aux_query_length
        aux_weight_i = max(r, 1)
        aux_weight_i = auxiliary_weight * math.log(aux_weight_i)
        aux_weight_i = 1 + max(aux_weight_i, 0)
    else:
        aux_weight_i = 0

    return aux_weight_i
