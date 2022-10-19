from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sized

import numpy as np
import tensorstore as ts
import torch
from datasets import Dataset
from loguru import logger
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import move_data_to_device
from torch import nn
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


@torch.inference_mode()
def cache_or_load_vectors(
    dataset: Dataset,
    model: Callable | nn.Module | LightningModule,
    *,
    trainer: Optional[Trainer] = None,
    model_output_key: Optional[str] = None,
    collate_fn: Optional[Pipe | Callable] = None,
    loader_kwargs: Optional[Dict] = None,
    cache_dir: Optional[str] = None,
    target_file: Optional[PathLike] = None,
    driver: str = "zarr",
    dtype: str = "float32",
) -> ts.TensorStore:
    """Process a `Dataset` using a model and cache the results into a `TensorStore`. If the cache
    already exists, it is loaded instead.

    Args:
        dataset (:obj:`Dataset`): Dataset to cache or load vectors.
        model (:obj:`Callable` | :obj:`nn.Module` | :obj:`LightningModule`): Model to use to
            compute the vectors.
        trainer (:obj:`Trainer`, optional): Trainer to use to compute the vectors.
        model_output_key (:obj:`str`, optional): List of keys to select from the model output
        collate_fn (:obj:`Pipe` | :obj:`Callable`, optional): Collate function for the DataLoader.
        loader_kwargs (:obj:`Dict`, optional): Keyword arguments for the DataLoader.
        cache_dir (:obj:`str`, optional): Directory to cache the vectors.
        target_file (:obj:`PathLike`, optional): Path to the target file.
        driver (:obj:`str`, optional): Driver to use for the TensorStore ("zarr" or "n5").
        dtype (:obj:`str`, optional): Data type to use for the TensorStore ("float16", "float32").

    Returns:
        ts.TensorStore: cached dataset vectors
    """
    # check the setup of the model and trainer
    if trainer is None:
        logger.warning("No Trainer was provided, setting up a default one.")
        trainer = Trainer()
    if not isinstance(model, LightningModule):
        model = LightningWrapper(model)
    msg = f"Model must be a LightningModule, got {type(model)}"
    assert isinstance(model, LightningModule), msg

    # infer the vector size from the model output
    dset_shape = _infer_dset_shape(
        dataset,
        model,
        model_output_key=model_output_key,
        collate_fn=collate_fn,
    )

    # define a unique fingerprint for the cache file
    fingerprint = get_fingerprint(
        {
            "model": get_fingerprint(model),
            "model_output_key": model_output_key,
            "dataset": dataset._fingerprint,
            "driver": driver,
            "dtype": dtype,
        }
    )

    # setup the cache file
    if target_file is None:
        assert cache_dir is not None, "cache_dir must be provided"
        target_file = Path(cache_dir) / "cached-vectors" / f"{fingerprint}.ts"
    target_file = Path(target_file)

    # make tensorstore config and init the store
    ts_config = make_ts_config(target_file, dset_shape, driver=driver, dtype=dtype)
    if not target_file.exists():
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
            output_key=model_output_key,
        )
        _process_dataset_with_lightning(
            dataset=dataset,
            model=model,
            tensorstore_callback=tensorstore_callback,
            trainer=trainer,
            collate_fn=collate_fn,
            loader_kwargs=loader_kwargs,
        )

    else:
        logger.info(f"Loading pre-computed vectors from {target_file.absolute()}")
        store = ts.open(ts_config, write=False, read=True)
        _validate_store(store, dset_shape, target_file)

    return store


async def load_store(
    path: PathLike,
    read: bool = True,
    write: bool = False,
    **kwargs,
) -> ts.TensorStore:
    """Load a TensorStore from a path."""
    path = Path(path)
    with open(path / "config.json", "r") as f:
        ts_config = json.load(f)
    store = await ts.open(ts_config, read=read, write=write, **kwargs)
    return store


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
