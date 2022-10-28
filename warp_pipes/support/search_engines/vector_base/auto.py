from __future__ import annotations

from typing import Dict

from loguru import logger

from .base import VectorBase
from .base import VectorBaseConfig
from .faiss import FaissVectorBase
from .torch import TorchVectorBase


def AutoVectorBase(config: Dict | VectorBaseConfig) -> VectorBase:
    if not isinstance(config, VectorBaseConfig):
        config = VectorBaseConfig(config)
    if config.index_factory == "torch":
        logger.info("Init TorchVectorBase")
        return TorchVectorBase(config)
    else:
        logger.info(
            "Init FaissVectorBase with index_factory: {}".format(config.index_factory)
        )
        return FaissVectorBase(config)
