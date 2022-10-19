from __future__ import annotations

import os.path
import shutil
import tempfile
from optparse import Option
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sized
from typing import Union

from omegaconf import DictConfig
from omegaconf import OmegaConf

try:
    from functools import singledispatchmethod
except Exception:
    from singledispatchmethod import singledispatchmethod

import pytorch_lightning as pl
import tensorstore as ts
import torch
from datasets import Dataset
from datasets import DatasetDict
from datasets import Split
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import move_data_to_device
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import SequentialSampler

from warp_pipes.pipes.collate import Collate
from warp_pipes.pipes.pipelines import Parallel
from warp_pipes.core.pipe import Pipe
from warp_pipes.pipes.basics import Identity
from warp_pipes.support.datastruct import Batch
from os import PathLike
from warp_pipes.support.fingerprint import get_fingerprint

# from warp_pipes.support.caching import cache_or_load_vectors
from warp_pipes.support.tensorstore_callback import select_field_from_output

from loguru import logger


class PredictWithoutCache(Pipe):
    """Process batches of data through a model and return the output."""

    def __init__(
        self,
        model: pl.LightningModule | nn.Module | Callable,
        **kwargs,
    ):
        super(PredictWithoutCache, self).__init__(**kwargs)
        self.model = model

    @torch.inference_mode()
    def _call_batch(
        self,
        batch: Batch,
        **kwargs,
    ) -> Batch:
        """Process a batch of data through the model and return the output."""
        if isinstance(self.model, nn.Module):
            device = next(iter(self.model.parameters())).device
            batch = move_data_to_device(batch, device)

        model_output = self.model(batch, **kwargs)
        return model_output

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls(Identity(), **kwargs)
