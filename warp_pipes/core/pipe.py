from __future__ import annotations

import json
import os
from abc import ABCMeta
from abc import abstractmethod
from copy import copy

try:
    from functools import singledispatchmethod
except Exception:
    from singledispatchmethod import singledispatchmethod

from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import T
from os import PathLike

import re
import stackprinter
import datasets
import jsondiff
import rich
from datasets import Dataset
from datasets import DatasetDict
from transformers import BatchEncoding

from warp_pipes.core.fingerprintable import Fingerprintable
from warp_pipes.core.condition import Condition
from warp_pipes.support.datastruct import Batch
from warp_pipes.support.datastruct import Eg
from warp_pipes.support.fingerprint import get_fingerprint
from warp_pipes.support.json_struct import reduce_json_struct
from warp_pipes.support.functional import get_batch_eg

from loguru import logger


class Pipe(Fingerprintable):
    """A pipe is a small unit of computation that ingests,
    modify and returns a batch of data."""

    __metaclass__ = ABCMeta
    # input filter applied to the input batch
    input_filter: Optional[Condition] = None
    # names of the keys required by the pipe
    requires_keys: Optional[List[str]] = None
    # whether the pipe allows setting `update=True``
    _allows_update: bool = True
    # whether the pipe allows setting a custom `input_filter`
    _allows_input_filter: bool = True
    # maximum number of parallel processes to use
    _max_num_proc: Optional[int] = None

    def __init__(
        self,
        *,
        input_filter: Optional[Condition] = None,
        update: bool = False,
        **kwargs,
    ):
        """
        Args:
            input_filter (:obj:`Condition`, optional) input key filter
            update (:obj:`bool`, optional) whether to update the input
                batch with the output of the pipe
        """
        super().__init__(**kwargs)
        if not self._allows_update and update:
            raise AttributeError(
                f"{type(self).__name__} does not allow using update=True"
            )

        if not self._allows_input_filter and input_filter is not None:
            raise AttributeError(
                f"{type(self).__name__} does not allow using input_filter"
            )

        if input_filter is not None:
            self.input_filter = input_filter
        self.update = update

    @singledispatchmethod
    def __call__(self, data: T, **kwargs) -> T:
        """
        Process the input data using the Pipe. The method is overloaded to handle different
        types of input data. Accepted types:
            - List[Eg]
            - Batch
            - datasets.Dataset
            - datasets.DatasetDict
        """
        raise TypeError(f"{type(self).__name__} does not support {type(data)}.")

    @__call__.register(datasets.arrow_dataset.Batch)
    @__call__.register(BatchEncoding)
    @__call__.register(dict)
    def _(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> Batch:
        """Apply the pipe to a batch of data. Potentially filter the keys using the input_filter.
        The output of `_call_batch()` is used to update the input batch (before filtering)
        if update=True, else the raw output is returned.
        """

        try:
            # filter some input keys
            _batch = self._filter_keys(batch)

            # process the batch
            output = self._call_batch(_batch, idx=idx, **kwargs)

            # update the input batch with the output if update is set to True
            if self.update:
                batch.update(output)
                output = batch

        except Exception as e:
            self.log_exception(e)
            raise e

        return output

    @__call__.register(list)
    def _(self, examples: List[Eg], idx: Optional[List[int]] = None, **kwargs) -> Batch:
        """Apply the pipe to a list of examples. Typically used to concatenate examples."""

        if not all(isinstance(eg, dict) for eg in examples):
            raise TypeError(
                f"Error in pipe {type(self)}, examples must be a list of dicts, "
                f"got {type(examples[0])}"
            )

        if self.update is True:
            raise AttributeError(
                "Pipe.update is set to True, cannot update a list of examples"
            )

        try:
            # filter some input keys
            _egs = list(map(self._filter_keys, examples))

            # process the batch
            output = self._call_egs(_egs, idx=idx, **kwargs)
        except Exception as e:
            self.log_exception(e)
            raise e

        return output

    @__call__.register(Dataset)
    def _(self, dataset: Dataset, **kwargs) -> Dataset:
        return self._call_dataset(dataset, **kwargs)

    @__call__.register(DatasetDict)
    def _(self, dataset: DatasetDict, **kwargs) -> DatasetDict:
        return self._call_dataset_dict(dataset, **kwargs)

    def _call_dataset_dict(self, dataset: DatasetDict, **kwargs) -> DatasetDict:
        """Process a `datasets.DatasetDict` using the Pipe"""
        new_datasets = {
            split: self._call_dataset(d, split=split, **kwargs)
            for split, d in dataset.items()
        }
        return DatasetDict(new_datasets)

    @abstractmethod
    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, **kwargs
    ) -> Batch:
        """Operation applied to the batch.

        Args:
          batch (:obj:`Batch`): input batch
            idx (:obj:`List[int]`, optional): indices of the examples in the batch

        Returns:
            :obj:`Batch`: output batch
        """
        raise NotImplementedError(f"_call_batch is not implemented for {type(self)}")

    @abstractmethod
    def _call_egs(
        self, examples: List[Eg], idx: Optional[List[int]] = None, **kwargs
    ) -> Batch:
        """Operation applied to a list of examples (Egs). Typically to concatenate examples."""
        raise NotImplementedError(f"_call_egs is not implemented for {type(self)}")

    def _call_dataset(
        self,
        dataset: Dataset,
        *,
        num_proc: int = 4,
        desc: Optional[str] = None,
        batch_size: int = 100,
        writer_batch_size: int = 1000,
        set_new_fingerprint: bool = False,
        keep_in_memory: bool = False,
        cache_fingerprint: Optional[PathLike] = None,
        fingerprint_kwargs_exclude: Optional[List[str]] = None,
        **kwargs,
    ) -> Dataset:
        """Process a `datasets.Dataset` using the Pipe.

        Args:
          dataset (:obj:`datasets.Dataset`): the dataset to process
          num_proc (:obj:`int`): number of parallel processes to use
          desc (:obj:`str`): description of the progress bar
          batch_size (:obj:`int`): batch size to use for each worker
          writer_batch_size (:obj:`int`): batch size to use for the PyArrow writer

        Returns:
            :obj:`datasets.Dataset`: the processed dataset
        """
        if fingerprint_kwargs_exclude is None:
            fingerprint_kwargs_exclude = []

        for key in ["batched", "with_indices"]:
            if key in kwargs.keys():
                raise ValueError(f"{key} cannot be set, it is always set as True.")

        if cache_fingerprint is not None:
            self._check_cached_fingerprint(
                cache_fingerprint,
                dataset,
                kwargs=kwargs,
                fingerprint_kwargs_exclude=fingerprint_kwargs_exclude,
            )

        if set_new_fingerprint:
            new_fingerprint_dict = {
                "dataset": dataset._fingerprint,
                "pipe": self.fingerprint(reduce=True),
                "params": {
                    k: get_fingerprint(v)
                    for k, v in kwargs.items()
                    if k not in fingerprint_kwargs_exclude
                },
            }
            new_fingerprint = get_fingerprint(new_fingerprint_dict)
            logger.info(
                f"{type(self).__name__}: Setting `new_fingerprint` to {new_fingerprint}"
            )
        else:
            new_fingerprint = None

        # clip the number of workers
        max_num_proc = self.max_num_proc
        if max_num_proc is not None:
            if num_proc > max_num_proc:
                logger.info(
                    f"{type(self).__name__}: Clipping number of workers to {max_num_proc}."
                )
                num_proc = max_num_proc

        if desc is None:
            desc = self.__class__.__name__

        # process the dataset using `Dataset.map`
        return dataset.map(
            self,
            num_proc=num_proc,
            desc=desc,
            batch_size=batch_size,
            batched=True,
            with_indices=True,
            writer_batch_size=writer_batch_size,
            new_fingerprint=new_fingerprint,
            keep_in_memory=keep_in_memory,
            fn_kwargs=kwargs,
        )

    @property
    def max_num_proc(self) -> Optional[int]:
        """Infer the maximum number of workers to use,
        check all children and takes the minimum value."""
        json_struct = self.to_json_struct(include_class_attributes=True)

        def safe_min(x):
            x = [y for y in x if isinstance(y, int)]
            if len(x):
                return min(x)
            else:
                return None

        def key_filter(key):
            return key == "_max_num_proc"

        return reduce_json_struct(
            json_struct, reduce_op=safe_min, key_filter=key_filter
        )

    def _filter_keys(self, batch: Batch) -> Batch:
        if self.input_filter is None:
            return batch

        return {k: v for k, v in batch.items() if self.input_filter(k)}

    def log_exception(self, e: Exception):
        log_dir = Path("warp-pipes.log")
        log_dir.mkdir(exist_ok=True, parents=True)
        log_file = log_dir / f"pipe-error-{type(self).__name__}-{os.getpid()}.log"
        logger.warning(
            f"Error in {type(self).__name__}. See full stack in {log_file.absolute()}"
        )
        with open(log_file, "w") as f:
            f.write(stackprinter.format())

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        """Instantiate a simple `Pipe` object for testing purposes."""
        raise NotImplementedError(
            f"`.instantiate_test()` is not implemented for {type(cls).__name__}"
        )

    @staticmethod
    def get_eg(
        batch: Batch, idx: int, filter_op: Optional[Callable] = None
    ) -> Dict[str, Any]:
        return get_batch_eg(batch=batch, idx=idx, filter_op=filter_op)

    def _check_cached_fingerprint(
        self,
        cache_dir: PathLike,
        dataset: Dataset,
        kwargs: Optional[Dict] = None,
        fingerprint_kwargs_exclude: Optional[List[str]] = None,
        debug: bool = True,
    ):
        """
        This method checks if the cached fingerprint is the same as the current one and save
        the current one. The cached fingerprint is based on the `Pipe.fingerprint()`,
        `Dataset._fingerprint` and `get_fingerprint(kwargs)`, kwargs matching
        `fingerprint_kwargs_exclude` are exlucded.

        TODO: remove this.
        """

        if cache_dir is None:
            logger.warning(
                "cache_dir is not provided, previous fingerprints cannot be verified."
            )
            return

        # kwargs exceptions
        if kwargs is not None:
            kwargs = copy(kwargs)
            if fingerprint_kwargs_exclude is not None:
                for key in fingerprint_kwargs_exclude:
                    kwargs.pop(key, None)

        # get a json-file from the current pipe
        fingerprints = self.fingerprint(reduce=False)
        fingerprints["__all__"] = self.fingerprint(reduce=True)

        # define the fingerprint for the kwargs
        kwargs_fingerprint_dict = {k: get_fingerprint(v) for k, v in kwargs.items()}
        # kwargs_fingerprint = get_fingerprint(kwargs_fingerprint_dict)
        fingerprints["__kwargs__"] = kwargs_fingerprint_dict

        # get the dataset fingerprint
        if isinstance(dataset, Dataset):
            dset_fingerprint = dataset._fingerprint
        elif isinstance(dataset, DatasetDict):
            dset_fingerprint = get_fingerprint(
                {k: d._fingerprint for k, d in dataset.items()}
            )
        else:
            raise TypeError(f"Cannot handle dataset type {type(dataset)}")
        fingerprints["__dataset__"] = dset_fingerprint

        # create the directory to store the fingerprints
        path = Path(cache_dir)
        if not path.exists():
            path.mkdir(parents=True)

        # compare to previously saved fingerprints
        name = self.__class__.__name__

        file = path / f"{name}-{dset_fingerprint}.json"
        if file.exists():
            prev_fingerprints = json.loads(file.read_text())
            diff = jsondiff.diff(prev_fingerprints, fingerprints)
            if len(diff) > 0:
                logger.warning(
                    f"Fingerprint for {name} changed from the latest run. "
                    f"Caching cannot be used. Enable debug logging mode to see the diff."
                )
                logger.debug(f"Fingerprint diff={diff}")
                if debug:
                    rich.print(f"[magenta] {name}: Fingerprints are different !")
                    rich.print(diff)
            else:
                logger.info(f"Fingerprint for {name} is identical to the latest run.")
        else:
            logger.info(f"No previous fingerprint found for {name}. file={file}")

        file.write_text(json.dumps(fingerprints, indent=2))
