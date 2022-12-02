from __future__ import annotations

import asyncio
import functools
import math
import os
import string
import subprocess
import time
from collections import defaultdict
from copy import copy
from typing import AsyncIterator
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

import torch
from elasticsearch import AsyncElasticsearch
from elasticsearch import Elasticsearch
from elasticsearch import helpers as es_helpers
from elasticsearch import RequestError
from loguru import logger
from pydantic import BaseModel

from warp_pipes.support.datastruct import Batch
from warp_pipes.support.datastruct import Eg
from warp_pipes.support.functional import iter_batch_egs


def compose_two_fns(f, g):
    return lambda *a, **kw: f(g(*a, **kw))


def compose_fns(*fs):
    return functools.reduce(compose_two_fns, fs)


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
    auxiliary_weight: float = 0
    scale_auxiliary_weight_by_lengths: bool = False
    output_keys: Optional[List[str]] = None


def es_search(
    batch: Batch,
    *,
    es_instance: Elasticsearch | AsyncElasticsearch,
    es_search_fn_config: EsSearchFnConfig,
    request_timeout: int = 600,
    chunk_size: Optional[int] = None,
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
    if (
        es_search_fn_config.auxiliary_weight > 0
        and es_search_fn_config.auxiliary_key not in batch
    ):
        raise ValueError(
            f"key `{es_search_fn_config.auxiliary_key}` must be provided if auxiliary_weight > 0, "
            f"Found keys {batch.keys()}"
        )

    def make_chunks_of_es_search_requests(
        batch: Batch, chunk_size: Optional[int]
    ) -> Iterable[List[List[Dict]]]:
        """Iterate over the batch and make groups of requests"""
        chunk = []
        idx = 0
        count = 0
        for eg in iter_batch_egs(batch):
            r = _make_query_request(eg, es_search_fn_config)
            count += 1
            chunk.extend(r)
            if chunk_size is not None and count == chunk_size:
                idx += 1
                yield chunk
                chunk = []
                count = 0

        if len(chunk):
            idx += 1
            yield chunk

    def es_msearch(es_requests: List[List[Dict]]):
        """Search and parse the results"""
        n_queries = len(es_requests) // 2
        fn = es_instance.msearch
        if isinstance(es_instance, AsyncElasticsearch):
            fn = compose_fns(asyncio.run, fn)
        response = fn(body=es_requests, request_timeout=request_timeout)
        n_responses = len(response["responses"])
        if n_responses != n_queries:
            raise ValueError(
                f"Expected {n_queries} responses, got {n_responses} responses."
            )
        return response

    def format_es_response(response: Dict) -> Batch:
        """Format the `msearch` response into a `Batch`"""
        # scan the results to identify the output keys
        output_keys = {
            "scores",
        }
        max_hits = 0
        for item_response in response["responses"]:
            if "hits" not in item_response:
                raise ValueError(
                    f"ES did not return any hit. Response: {item_response}"
                )
            max_hits = max(max_hits, len(item_response["hits"]["hits"]))
            for hit in item_response["hits"]["hits"][:1]:
                output_keys |= set(hit["_source"].keys())

        # traverse all responses to create the batch of results
        n_responses = len(response["responses"])
        batch_of_results = {
            k: [list() for _ in range(n_responses)] for k in output_keys
        }
        for j, item_response in enumerate(response["responses"]):
            for hit in item_response["hits"]["hits"]:
                hit_data = {"scores": hit["_score"], **hit["_source"]}
                for k, v in hit_data.items():
                    batch_of_results[k][j].append(v)

        return batch_of_results

    def search_by_chunks(batch: Batch, chunk_size: int) -> Batch:
        """Search all the requests in the batch"""
        idx = 0
        all_results = defaultdict(list)
        for es_requests in make_chunks_of_es_search_requests(batch, chunk_size):
            idx += 1

            r = es_msearch(es_requests)
            r = format_es_response(r)
            for key, value in r.items():
                all_results[key].extend(value)

        return all_results

    return search_by_chunks(batch, chunk_size)


def _make_query_request(eg, config: EsSearchFnConfig) -> List[Dict]:
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


def es_create_index(
    index_name: str,
    *,
    es_instance: Elasticsearch | AsyncElasticsearch,
    body: Optional[Dict] = None,
) -> bool:
    try:
        fn = es_instance.indices.create
        if isinstance(es_instance, AsyncElasticsearch):
            fn = compose_fns(asyncio.run, fn)
        response = fn(index=index_name, body=body)
        logger.info(response)
        newly_created = True

    except RequestError as err:
        if err.error == "resource_already_exists_exception":
            newly_created = False
        else:
            raise err

    return newly_created


def es_remove_index(
    index_name: str, *, es_instance: Elasticsearch | AsyncElasticsearch
):
    fn = es_instance.indices.delete
    if isinstance(es_instance, AsyncElasticsearch):
        fn = compose_fns(asyncio.run, fn)
    return fn(index=index_name)


def es_ingest(
    batch: Batch,
    *,
    es_instance: Elasticsearch,
    index_name: str,
    chunk_size=1000,
    request_timeout=200,
):
    def gen_actions(batch, index_name):
        for eg in iter_batch_egs(batch):
            yield {"_index": index_name, "_source": eg}

    return es_helpers.bulk(
        es_instance,
        gen_actions(batch, index_name),
        chunk_size=chunk_size,
        request_timeout=request_timeout,
        refresh="true",
    )


async def async_es_ingest(
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
        raise_on_error=True,
    ):
        action, result = result.popitem()
        if not ok:
            raise ValueError(f"Failed to {action}: {result}")


def ping_es(host=None, **kwargs):
    if host is None:
        hosts = None
    else:
        hosts = [host]
    return Elasticsearch(hosts=hosts).ping(**kwargs)


class ElasticSearchInstance(object):
    """Instantiate ElasticSearch in a subprocess."""

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
