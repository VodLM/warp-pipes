import math
import os
import string
import subprocess
import time
from collections import defaultdict
from copy import copy
from typing import Dict
from typing import List
from typing import Optional

import torch
from elasticsearch import Elasticsearch
from elasticsearch import helpers as es_helpers
from elasticsearch import RequestError
from loguru import logger

from warp_pipes.support.datastruct import Batch
from warp_pipes.support.functional import iter_batch_egs


def es_search(
    batch: Batch,
    *,
    es_instance: Elasticsearch,
    index_name: str,
    query_key: str,
    auxiliary_key: Optional[str] = None,
    filter_key: Optional[str] = None,
    auxiliary_weight: float = 0,
    scale_auxiliary_weight_by_lengths: bool = False,
    k: int = 10,
    request_timeout: int = 600,
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
    With filtering, only documents nmatching `batch[filter_key]` on the field `filter_key` are
    returned.

    Args:
        batch (:obj:`Batch`): input query
        es_instance (:obj:`Elasticsearch`): Elasticsearch instance
        index_name (:obj:`str`): name of the elasticsearch index
        query_key (:obj:`str`): name of the elasticsearch field to query and name of the key in the
            batch
        auxiliary_key (:obj:`str`): name of the batch field used to query the main elasticsearch
            field (`query_key`)
        filter_key (:obj:`str`): name of the field used for filtering
        auxiliary_weight (:obj:`float`): value of the weight (beta)
        scale_auxiliary_weight_by_lengths (:obj:`bool`): scale the auxiliary weight by the ratio
            of the lengths `query`  / `aux_query`
        k (:obj:`int`): number of results
        request_timeout (:obj:`int`): timeout value
    """
    if auxiliary_weight > 0 and auxiliary_key not in batch:
        raise ValueError(
            f"key `{auxiliary_key}` must be provided if auxiliary_weight > 0, "
            f"Found keys {batch.keys()}"
        )

    request = []
    for eg in iter_batch_egs(batch):
        use_aux_queries = auxiliary_weight > 0
        should_query_parts = []
        filter_query_parts = []

        # make the main query
        query = eg[query_key]
        should_query_parts.append(
            {
                "match": {
                    query_key: {
                        "query": query,
                        "operator": "or",
                    }
                }
            },
        )

        # make the auxiliary query
        if use_aux_queries:
            aux_query = eg[auxiliary_key]
            if scale_auxiliary_weight_by_lengths:
                aux_weight_i = _get_scaled_auxiliary_weight(
                    auxiliary_weight, query, aux_query
                )
            else:
                aux_weight_i = auxiliary_weight

            should_query_parts.append(
                {
                    "match": {
                        query_key: {
                            "query": aux_query,
                            "operator": "or",
                            "boost": aux_weight_i,
                        }
                    }
                },
            )

        # make the filter query
        if filter_key is not None:
            filter_query = eg[filter_key]
            if isinstance(filter_query, torch.Tensor):
                filter_query = filter_query.item()
            filter_query_parts.append({"term": {filter_key: filter_query}})

        # make the final request
        r = {
            "query": {
                "bool": {"should": should_query_parts, "filter": filter_query_parts},
            },
            "from": 0,
            "size": k,
        }

        # append the header and body of the request
        request.extend([{"index": index_name}, r])

    # run the search
    result = es_instance.msearch(
        body=request, index=index_name, request_timeout=request_timeout
    )

    results = defaultdict(list)
    for response in result["responses"]:
        if "hits" not in response:
            raise ValueError(f"ES did not return any hits. Response: {response}")

        result_i = defaultdict(list)
        for hit in response["hits"]["hits"]:
            hit_data = {"scores": hit["_score"], **hit["_source"]}
            for k, v in hit_data.items():
                result_i[k].append(v)

        for k, v in result_i.items():
            results[k].append(v)

    return results


def es_create_index(
    index_name: str, *, es_instance: Elasticsearch, body: Optional[Dict] = None
) -> bool:
    try:
        response = es_instance.indices.create(index=index_name, body=body)
        logger.info(response)
        newly_created = True

    except RequestError as err:
        if err.error == "resource_already_exists_exception":
            newly_created = False
        else:
            raise err

    return newly_created


def es_remove_index(index_name: str, *, es_instance: Elasticsearch):
    return es_instance.indices.delete(index=index_name)


def es_ingest(
    batch: Batch,
    *,
    es_instance: Elasticsearch,
    index_name: str,
    chunk_size=1000,
    request_timeout=200,
):
    actions = []
    for eg in iter_batch_egs(batch):
        actions.append(
            {
                "_index": index_name,
                "_source": eg,
            }
        )

    return es_helpers.bulk(
        es_instance,
        actions,
        chunk_size=chunk_size,
        request_timeout=request_timeout,
        refresh="true",
    )


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
