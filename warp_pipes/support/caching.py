from __future__ import annotations

import json
import os
from copy import copy
from os import PathLike
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Sized
from typing import Union

import numpy as np
import pydantic
import tensorstore as ts
import torch
from datasets import Dataset
from loguru import logger
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import move_data_to_device
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from warp_pipes.core.pipe import Pipe
from warp_pipes.pipes.collate import Collate
from warp_pipes.pipes.pipelines import Parallel
from warp_pipes.support.datastruct import Batch
from warp_pipes.support.fingerprint import get_fingerprint
from warp_pipes.support.tensorstore_callback import IDX_COL
from warp_pipes.support.tensorstore_callback import select_key_from_output
from warp_pipes.support.tensorstore_callback import TensorStoreCallback

DEFAULT_LOADER_KWARGS = {"batch_size": 10, "num_workers": 2, "pin_memory": True}
PREDICT_VECTOR_NAME = "vector"


def make_ts_config(
    path: PathLike,
    dset_shape: List[int],
    chunk_size: int = 100,
    driver: str = "zarr",
    dtype: str = "float32",
):
    driver_meta = {
        "n5": {
            "dataType": dtype,
            "dimensions": dset_shape,
            "compression": {"type": "gzip"},
            "blockSize": [chunk_size, *dset_shape[1:]],
        },
        "zarr": {
            "dtype": {"float16": "<f2", "float32": "<f4", "float64": "<f8"}[dtype],
            "shape": dset_shape,
            "chunks": [chunk_size, *dset_shape[1:]],
        },
    }

    return {
        "driver": driver,
        "kvstore": {
            "driver": "file",
            "path": str(path),
        },
        "metadata": {
            **driver_meta[driver],
        },
    }


class CacheConfig(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True

    cache_dir: Optional[Path] = None
    trainer_: Optional[Trainer] = pydantic.Field(alias="trainer")
    collate_fn_: Optional[Union[Pipe, Callable]] = pydantic.Field(alias="collate_fn")
    loader_kwargs_: Optional[Dict] = pydantic.Field(alias="loader_kwargs")
    model_output_key: Optional[str]
    cache_dir: Path
    driver: Literal["zarr", "n5"] = "zarr"
    dtype: Literal["float16", "float32"] = "float32"

    @property
    def trainer(self):
        return self.trainer_ or Trainer()

    @property
    def collate_fn(self):
        return self.collate_fn_ or Collate()

    @property
    def loader_kwargs(self):
        return self.loader_kwargs_ or copy(DEFAULT_LOADER_KWARGS)


def maybe_wrap_model(model):
    if not isinstance(model, LightningModule):
        model = LightningWrapper(model)
    if not isinstance(model, LightningModule):
        raise TypeError(f"Model must be a LightningModule, got {type(model)}")
    return model


@torch.inference_mode()
def cache_or_load_vectors(
    dataset: Dataset,
    model: Callable | nn.Module | LightningModule,
    *,
    cache_dir: Path,
    config: Dict | CacheConfig,
) -> ts.TensorStore:
    """Process a `Dataset` using a model and cache the results into a `TensorStore`. If the cache
    already exists, it is loaded instead.

    Args:
        dataset (:obj:`Dataset`): Dataset to cache or load vectors.
        model (:obj:`Callable` | :obj:`nn.Module` | :obj:`LightningModule`): Model to use to
            compute the vectors.
       config (:obj:`Dict` | :obj:`CacheConfig`, `optional`): Configuration for the caching

    Returns:
        ts.TensorStore: cached dataset vectors
    """
    if not isinstance(config, CacheConfig):
        config = CacheConfig(**config)

    model = maybe_wrap_model(model)

    # infer the vector size from the model output
    dset_shape = _infer_dset_shape(
        dataset,
        model,
        model_output_key=config.model_output_key,
        collate_fn=config.collate_fn,
    )

    # define a unique fingerprint for the cache file
    fingerprint = get_fingerprint(
        {
            "model": get_fingerprint(model),
            "model_output_key": config.model_output_key,
            "dataset": dataset._fingerprint,
            "driver": config.driver,
            "dtype": config.dtype,
        }
    )

    # setup the cache file
    assert cache_dir is not None, "cache_dir must be provided"
    target_file = Path(cache_dir) / "cached-vectors" / f"{fingerprint}.ts"

    # make tensorstore config and init the store
    ts_config = make_ts_config(
        target_file, dset_shape, driver=config.driver, dtype=config.dtype
    )
    # synchronize all workers before checking if the file exists
    # dist.barrier()

    if not target_file.exists():
        # only the first worker creates the file
        if dist.get_rank() == 0:
            logger.info(f"Writing vectors to {target_file.absolute()}")
            store = ts.open(ts_config, create=True, delete_existing=False).result()
            with open(target_file / "config.json", "w") as f:
                ts_config_ = {
                    k: v for k, v in ts_config.items() if k in ["driver", "kvstore"]
                }
                json.dump(ts_config_, f)

        # init a callback to store predictions in the TensorStore
        tensorstore_callback = TensorStoreCallback(
            store=store,
            output_key=config.model_output_key,
            asynch=True,
        )
        _process_dataset_with_lightning(
            dataset=dataset,
            model=model,
            tensorstore_callback=tensorstore_callback,
            trainer=config.trainer,
            collate_fn=config.collate_fn,
            loader_kwargs=config.loader_kwargs,
        )
        futures = tensorstore_callback.futures

        # make sure all writes are complete
        for future in futures:
            future.result()

        # close the store
        del store
    else:
        logger.info(f"Loading pre-computed vectors from {target_file.absolute()}")

    # synchronize all workers after checking if the file exists
    # dist.barrier()

    # reload the same TensorStore in read mode
    store = load_store(target_file, read=True, write=False)
    _validate_store(store, dset_shape, target_file)

    return store


def load_store(
    path: PathLike | Dict,
    read: bool = True,
    write: bool = False,
    use_pdb: bool = False,
    **kwargs,
) -> ts.TensorStore:
    """Load a TensorStore from a path."""
    if isinstance(path, dict):
        ts_config = path
    else:
        path = Path(path)
        with open(path / "config.json", "r") as f:
            ts_config = json.load(f)

    logger.info(f"Loading store (pid={os.getpid()}) {ts_config}...")
    if use_pdb:
        import remote_pdb

        remote_pdb.set_trace()
    return ts.open(ts_config, read=read, write=write, **kwargs).result()


def load_store_test_(c):
    store = load_store(c)
    return store.shape


def infer_path_from_store(store: ts.TensorStore) -> Path:
    path = store.spec().kvstore.path
    return Path(path)


def _validate_store(
    store: ts.TensorStore,
    dset_shape: List[int],
    target_file: PathLike,
    chunk_size: int = 1_000,
):
    # check that the store has the correct shape
    if list(store.shape) != dset_shape:
        raise ValueError(
            f"Dataset of shape={dset_shape} not matching "
            f"the store with shape={list(store.shape)}. "
            f"Consider deleting and re-computing these vectors.\n"
            f"path={target_file.absolute()}"
        )

    # check that all values are set
    for idx in tqdm(range(0, dset_shape[0], chunk_size), desc="Validating store"):
        vectors = store[idx : min(idx + chunk_size, dset_shape[0])].read().result()
        if np.absolute(vectors).sum() == 0:
            raise ValueError(
                f"Store is missing values in the range [{idx}, {idx + chunk_size}]. "
                f"Consider deleting and re-computing these vectors.\n"
                f"path={target_file.absolute()}"
            )


def _process_dataset_with_lightning(
    dataset: Dataset,
    model: LightningModule,
    *,
    tensorstore_callback: TensorStoreCallback,
    trainer: Optional[Trainer],
    collate_fn: Optional[Callable],
    loader_kwargs: Optional[Dict],
):
    """
    Process the dataset using the Trainer an the model.
    """
    trainer.callbacks.append(tensorstore_callback)

    # init the dataloader (the collate_fn and dataset are wrapped to return the ROW_IDX)
    loader = _init_loader(dataset, collate_fn=collate_fn, loader_kwargs=loader_kwargs)

    # run the trainer predict method, model.forward() is called
    # for each batch and store into the callback cache
    trainer.predict(model=model, dataloaders=loader, return_predictions=False)
    trainer.callbacks.remove(tensorstore_callback)
    del loader


class LightningWrapper(LightningModule):
    def __init__(self, model: Callable):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        return self.forward(batch)

    def validation_step(self, batch, batch_idx):
        return self.forward(batch)

    def test_step(self, batch, batch_idx):
        return self.forward(batch)


class AddRowIdx(TorchDataset):
    """This class is used to add the column `IDX_COL` to the batch"""

    def __init__(self, dataset: Sized):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item) -> Batch:
        batch = self.dataset[item]
        batch[IDX_COL] = item
        return batch


def _init_loader(
    dataset: Dataset,
    collate_fn: Optional[Callable] = None,
    loader_kwargs: Optional[Dict] = None,
    wrap_indices: bool = True,
) -> DataLoader:
    if collate_fn is None:
        collate_fn = Collate()

    if wrap_indices:
        dataset = _wrap_dataset(dataset)
        collate_fn = _wrap_collate_fn(collate_fn)

    loader_kwargs = loader_kwargs or DEFAULT_LOADER_KWARGS
    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        shuffle=False,
        persistent_workers=False,
        **loader_kwargs,
    )
    return loader


def _wrap_collate_fn(collate_fn: Callable) -> Pipe:
    """Wrap the collate_fn to return IDX_COL along the batch values"""
    return Parallel(collate_fn, Collate(IDX_COL))


def _wrap_dataset(dataset: Dataset) -> TorchDataset:
    """Wrap the dataset to return IDX_COL along the batch values"""
    return AddRowIdx(dataset)


def _infer_dset_shape(
    dataset: Dataset,
    model: LightningModule,
    collate_fn: Callable,
    model_output_key: Optional[str],
) -> List[int]:
    """
    Infer the dataset shape by running the model on a single batch.
    """
    batch = collate_fn([dataset[0]])
    batch = move_data_to_device(batch, model.device)
    output = model(batch)
    vector_shape = select_key_from_output(output, model_output_key).shape[1:]
    return [len(dataset), *vector_shape]
