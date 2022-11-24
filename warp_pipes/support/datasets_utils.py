from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import datasets
import numpy as np
from datasets import Dataset
from datasets import DatasetDict
from datasets import Split
from omegaconf import DictConfig
from omegaconf import ListConfig

from warp_pipes.core.condition import Condition
from warp_pipes.support.fingerprint import get_fingerprint

HfDataset = Union[Dataset, DatasetDict]


def take_subset(
    dataset: HfDataset, subset_size: float | int | Dict[Split, float] | Dict[Split, int]
) -> HfDataset:
    """Take a subset of the dataset and return."""
    if isinstance(dataset, Dataset):
        if not isinstance(subset_size, (float, int)):
            raise ValueError(
                f"subset_size must be a float or int, got {type(subset_size)}"
            )

        # take a fraction of the dataset
        if isinstance(subset_size, float) and 0 <= subset_size <= 1:
            subset_size = int(subset_size * len(dataset))

        if len(dataset) < subset_size:
            return dataset

        # select subset and return
        rgn = np.random.RandomState(0)
        indices = rgn.choice(len(dataset), subset_size, replace=False)
        subset_dataset = dataset.select(indices)
        return subset_dataset

    elif isinstance(dataset, DatasetDict):
        if isinstance(subset_size, (float, int)):
            subset_size = {split: subset_size for split in dataset.keys()}

        elif not isinstance(subset_size, (DatasetDict, DictConfig)):
            raise TypeError(
                f"subset_size must be a float, int or dict, " f"got {type(subset_size)}"
            )
        return DatasetDict(
            {
                split: take_subset(ds, subset_size[split])
                for split, ds in dataset.items()
            }
        )
    else:
        raise TypeError(
            f"dataset must be a Dataset or DatasetDict, got {type(dataset)}"
        )


def format_size_difference(
    original_size: Dict[str, int], new_dataset: DatasetDict
) -> str:
    # store the previous split sizes
    prev_lengths = {k: v for k, v in original_size.items()}
    new_lengths = {k: len(v) for k, v in new_dataset.items()}
    u = "Dataset size after filtering ("
    for key in new_lengths.keys():
        ratio = new_lengths[key] / prev_lengths[key]
        u += f"{key}: {new_lengths[key]} ({100 * ratio:.0f}%), "
    return u + ")"


def get_column_names(dataset: HfDataset) -> List[str]:
    if isinstance(dataset, DatasetDict):
        names = [c for dset in dataset.values() for c in dset.column_names]
        return list(set(names))
    elif isinstance(dataset, Dataset):
        return dataset.column_names
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def remove_columns_with_fingerprint(dataset: Dataset, columns: List[str]) -> Dataset:
    """Remove columns and set a fingerprint deterministically"""
    columns = list(sorted(columns))
    args = {"dataset": dataset._fingerprint, "columns": columns}
    new_fingerprint = get_fingerprint(args)
    return dataset.remove_columns(columns, new_fingerprint=new_fingerprint)


def keep_only_columns(
    dataset: HfDataset, columns: Optional[List[str] | Condition]
) -> HfDataset:
    """Keep only the given columns and set a fingerprint deterministically"""
    if columns is None:
        return dataset
    else:
        if isinstance(columns, (list, set, ListConfig)):
            cols_to_drop = [c for c in get_column_names(dataset) if c not in columns]
        elif isinstance(columns, Condition):
            cols_to_drop = [c for c in get_column_names(dataset) if not columns(c)]
        else:
            raise ValueError(f"Unsupported columns type: {type(columns)}")

        dataset = remove_columns(dataset, cols_to_drop)

        return dataset


def remove_columns(dataset: HfDataset, cols_to_drop: List[str]) -> HfDataset:
    if isinstance(dataset, Dataset):
        dataset = remove_columns_with_fingerprint(dataset, cols_to_drop)
    elif isinstance(dataset, DatasetDict):
        dataset = DatasetDict(
            {
                k: remove_columns_with_fingerprint(v, cols_to_drop)
                for k, v in dataset.items()
            }
        )
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")
    return dataset


def concatenate_datasets(dsets: List[HfDataset], **kwargs) -> HfDataset:
    types = [type(ds) for ds in dsets]
    if len(set(types)) > 1:
        raise ValueError(f"All datasets must be of the same type. Found {types}")
    dset_type = types[0]
    if dset_type == Dataset:
        dataset = datasets.concatenate_datasets(dsets, **kwargs)
    elif dset_type == DatasetDict:
        keys = set.union(*[set(ds.keys()) for ds in dsets])
        dataset = DatasetDict(
            {
                k: datasets.concatenate_datasets([dset[k] for dset in dsets], **kwargs)
                for k in keys
            }
        )
    else:
        raise TypeError(f"Unsupported dataset type: {dset_type}")
    return dataset


def get_dataset_fingerprints(
    dataset: HfDataset, reduce: bool = False
) -> Dict[str, str] | str:
    """Fingerprint a `HfDataset`"""
    if isinstance(dataset, Dataset):
        fingerprint_state = dataset._fingerprint
    elif isinstance(dataset, DatasetDict):
        fingerprint_state = {k: v._fingerprint for k, v in dataset.items()}
    else:
        raise ValueError("Unsupported dataset type")

    if reduce and not isinstance(fingerprint_state, str):
        fingerprint_state = get_fingerprint(fingerprint_state)

    return fingerprint_state
