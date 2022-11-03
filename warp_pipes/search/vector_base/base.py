from __future__ import annotations

import abc
from os import PathLike
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import faiss
import pydantic
import torch
from pydantic import BaseModel

from warp_pipes.core.fingerprintable import Fingerprintable
from warp_pipes.search.vector_base.utils.faiss import FaissFactory


class VectorBaseConfig(BaseModel, Fingerprintable):
    index_factory: str = "IVF100"
    dimension: Optional[int] = None  # can be set at runtime
    nprobe: int = 32
    faiss_metric: Union[str, int] = faiss.METRIC_INNER_PRODUCT
    train_on_cpu: bool = False
    keep_on_cpu: bool = False
    tempmem = -1
    use_float16: bool = True
    max_add_per_gpu: int = 100_000
    add_batch_size: int = 65536
    use_precomputed_tables: bool = False
    replicas: int = 1
    shard: bool = False
    train_size: Optional[int] = None
    random_train_subset: bool = False

    @pydantic.validator("faiss_metric")
    def infer_faiss_metric(cls, v):
        metric_lookup = {
            "inner_product": faiss.METRIC_INNER_PRODUCT,
            "l2": faiss.METRIC_L2,
        }
        metric_lookup.update({str(v): v for v in metric_lookup.values()})
        if isinstance(v, str):
            v = metric_lookup[v]
        if not isinstance(v, int):
            raise TypeError(f"faiss_metric must be an int, not {type(v)}")
        return v

    @property
    def factory(self) -> FaissFactory:
        return FaissFactory(self.index_factory)


class VectorBase:
    """A class to handle the indexing of vectors."""

    __metaclass__ = abc.ABCMeta
    _config_type: type = VectorBaseConfig

    def __init__(self, config: Dict | VectorBaseConfig):
        if not isinstance(config, VectorBaseConfig):
            config = self._config_type(config)
        self.config = config

    @abc.abstractmethod
    def train(self, vectors, **kwargs):
        ...

    @abc.abstractmethod
    def add(self, vectors: torch.Tensor, **kwargs):
        ...

    @abc.abstractmethod
    def save(self, path: PathLike):
        ...

    @abc.abstractmethod
    def load(self, path: PathLike):
        ...

    @abc.abstractmethod
    def search(
        self, query: torch.Tensor, k: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @property
    @abc.abstractmethod
    def ntotal(self) -> int:
        ...

    @abc.abstractmethod
    def cuda(self, devices: Optional[List[int]] = None):
        ...

    @abc.abstractmethod
    def cpu(self):
        ...
