from __future__ import annotations

import json
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

import re
import datasets
import jsondiff
import rich
from datasets import Dataset
from datasets import DatasetDict
from transformers import BatchEncoding

from fz_openqa.datamodules.component import Component
from fz_openqa.datamodules.pipes.control.condition import Condition
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import Eg
from fz_openqa.utils.datastruct import PathLike
from fz_openqa.utils.fingerprint import get_fingerprint
from fz_openqa.utils.functional import get_batch_eg
from fz_openqa.utils.json_struct import reduce_json_struct

from loguru import logger


def slice_batch(batch: Batch, i: int | slice) -> Batch:
    """

    Args:
      batch: Batch:
      i: int | slice:

    Returns:


    """
    return {k: v[i] for k, v in batch.items()}


def camel_to_snake(name):
    """

    Args:
      name:

    Returns:


    """
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


class Pipe(Component):
    """A pipe is a small unit of computation that ingests,
    modify and returns a batch of data.

    ----------
    Attributes
    id
       An identifier for the pipe.
    input_filter
        Condition used to filter keys in the input data.
    update
        If set to True, output the input batch with the output batch.
    requires_keys
       A list of keys that the pipe requires to be present in the data.

    Args:

    Returns:


    """

    # __metaclass__ = ABCMeta
    id: Optional[str] = None
    input_filter: Optional[Condition] = None
    requires_keys: Optional[List[str]] = None
    _allows_update: bool = True
    _allows_input_filter: bool = True
    _backend: Optional[str] = None
    _max_num_proc: Optional[int] = None

    def __init__(
        self,
        *,
        id: Optional[str] = None,
        input_filter: Optional[Condition] = None,
        update: bool = False,
    ):
        """
        Parameters
        ----------
        id
           An identifier for the pipe.
        input_filter
            a condition used to filter keys in the input data
            (keys that do not satisfy the condition are removed)
        update
            If set to True, output the input batch updated with the output batch.
        """
        super().__init__(id=id)
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

    def output_keys(self, input_keys: List[str]) -> List[str]:
        """Return the list of keys that the pipe is expected to return.

        Args:
          input_keys: The list of keys that the pipe expects as input.
          input_keys: List[str]:

        Returns:


        """
        output_keys = copy(input_keys)
        if self.input_filter is not None:
            output_keys = list(
                {k: None for k in output_keys if self.input_filter(k)}.keys()
            )

        if self.update:
            output_keys = input_keys + output_keys

        return output_keys

    @staticmethod
    def get_eg(
        batch: Batch, idx: int, filter_op: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Extract example `idx` from a batch, potentially filter keys.

        Args:
          batch: Input batch
          idx: Index of the example to extract
          filter_op: A function that used to filter the keys
          batch: Batch:
          idx: int:
          filter_op: Optional[Callable]:  (Default value = None)

        Returns:


        """
        return get_batch_eg(batch=batch, idx=idx, filter_op=filter_op)

    @singledispatchmethod
    def __call__(
        self,
        data: T,
        idx: Optional[List[int]] = None,
        num_proc: int = 4,
        desc: Optional[str] = None,
        batch_size: Optional[int] = None,
        writer_batch_size: Optional[int] = None,
        set_new_fingerprint: bool = False,
        **kwargs,
    ) -> T:
        """
        Apply the pipe to a data. Potentially filter the keys using the input_filter.
        This method is dispatched on the type of the input data.

        Parameters
        ----------
        data
            The input data
        idx
            indexes of the batch examples
        num_proc
            For `Dataset` input only: number of processes to use
        desc
            For `Dataset` input only: description for the progress bar
        writer_batch_size
            For `Dataset` input only: batch size for the pyarrow writer
        set_new_fingerprint
            For `Dataset` input only: set `new_fingerprint` using `Pipe.get_fingerprint`.
        kwargs
            additional arguments

        Returns
        -------
        Batch
            The output data
        """

        raise TypeError(f"{type(self).__name__} does not support {type(data)}.")

    @__call__.register(datasets.arrow_dataset.Batch)
    @__call__.register(BatchEncoding)
    @__call__.register(dict)
    def _(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> Batch:
        """Apply the pipe to a batch of data. Potentially filter the keys using the input_filter.
        The output of `_call_batch()` is used to update the input batch (before filtering)
        if update=True, else the raw output is returned.

        Args:
          batch: batch to apply the pipe to
          idx: indexes of the batch examples
          kwargs: additional arguments
          batch: Batch:
          idx: Optional[List[int]]:  (Default value = None)
          **kwargs:

        Returns:


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
            logger.exception(e)
            raise e

        return output

    @__call__.register(list)
    def _(self, examples: List[Eg], idx: Optional[List[int]] = None, **kwargs) -> Batch:
        """Apply the pipe to a list of examples. Typically to concatenate examples.

        Args:
          examples: batch of examples to apply the pipe to
          idx: indexes of the examples
          kwargs: additional arguments
          examples: List[Eg]:
          idx: Optional[List[int]]:  (Default value = None)
          **kwargs:

        Returns:


        """

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
            logger.exception(e)
            raise e

        return output

    @__call__.register(Dataset)
    def _(
        self,
        dataset: Dataset,
        *,
        num_proc: int = 4,
        desc: Optional[str] = None,
        batch_size: Optional[int] = None,
        writer_batch_size: Optional[int] = None,
        **kwargs,
    ) -> Dataset:
        """Apply the Pipe to a `Dataset`

        Args:
          dataset: A Huggingface Dataset object
          num_proc: Number of workers
          desc: Description for the progress bar
          batch_size: Batch size for each worker
          writer_batch_size: Batch size for the pyarrow writer
          kwargs:
          dataset: Dataset:
          *:
          num_proc: int:  (Default value = 4)
          desc: Optional[str]:  (Default value = None)
          batch_size: Optional[int]:  (Default value = None)
          writer_batch_size: Optional[int]:  (Default value = None)
          **kwargs:

        Returns:


        """
        return self._call_dataset(
            dataset,
            num_proc=num_proc,
            desc=desc,
            batch_size=batch_size,
            writer_batch_size=writer_batch_size,
            **kwargs,
        )

    @__call__.register(DatasetDict)
    def _(self, dataset: DatasetDict, **kwargs) -> DatasetDict:
        """Apply the Pipe to a `DatasetDict`

        Args:
          dataset: A Huggingface DatasetDict object
          kwargs:
          dataset: DatasetDict:
          **kwargs:

        Returns:


        """
        return self._call_dataset_dict(dataset, **kwargs)

    def _call_dataset_dict(self, dataset: DatasetDict, **kwargs) -> DatasetDict:
        """

        Args:
          dataset: DatasetDict:
          **kwargs:

        Returns:


        """
        new_datasets = {
            split: self._call_dataset(d, split=split, **kwargs)
            for split, d in dataset.items()
        }
        return DatasetDict(new_datasets)

    @abstractmethod
    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, **kwargs
    ) -> Batch:
        """Main operation applied to the batch.

        Args:
          batch: batch to apply the pipe to
          idx: indexes of the batch examples
          kwargs: additional arguments
          batch: Batch:
          idx: Optional[List[int]]:  (Default value = None)
          **kwargs:

        Returns:


        """
        raise NotImplementedError(f"_call_batch is not implemented for {type(self)}")

    @abstractmethod
    def _call_egs(
        self, examples: List[Eg], idx: Optional[List[int]] = None, **kwargs
    ) -> Batch:
        """Main Operation applied to a list of examples (Egs). Typically to concatenate examples.

        Args:
          examples: List of examples
          idx: indexes of the examples
          kwargs: additional arguments
          examples: List[Eg]:
          idx: Optional[List[int]]:  (Default value = None)
          **kwargs:

        Returns:


        """
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
        """Apply the Pipe to a `Dataset`

        Args:
          dataset: A Huggingface Dataset object
          num_proc: Number of workers
          desc: Description for the progress bar
          batch_size: Batch size for each worker
          writer_batch_size: Batch size for the pyarrow writer
          kwargs: Additional attributes passed to the pipe
          set_new_fingerprint: If True, the `new_fingerprint` will de defined
        using `Pipe.fingerprint()` and `Dataset._fingerprint`
          cache_fingerprint: If set to a path, the fingerprint will be cache to that directory.
          dataset: Dataset:
          *:
          num_proc: int:  (Default value = 4)
          desc: Optional[str]:  (Default value = None)
          batch_size: int:  (Default value = 100)
          writer_batch_size: int:  (Default value = 1000)
          set_new_fingerprint: bool:  (Default value = False)
          keep_in_memory: bool:  (Default value = False)
          cache_fingerprint: Optional[PathLike]:  (Default value = None)
          fingerprint_kwargs_exclude: Optional[List[str]]:  (Default value = None)
          **kwargs:

        Returns:


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
        """Maximum number of workers to use, check all children and takes the minimum value

        todo: investigate why the values needs to be filtered by type
         (type of `_max_num_proc`: `int` returned as well)

        Args:

        Returns:

        """
        json_struct = self.to_json_struct(include_class_attributes=True)

        def safe_min(x):
            """

            Args:
              x:

            Returns:


            """
            x = [y for y in x if isinstance(y, int)]
            if len(x):
                return min(x)
            else:
                return None

        def key_filter(key):
            """

            Args:
              key:

            Returns:


            """
            return key == "_max_num_proc"

        return reduce_json_struct(
            json_struct, reduce_op=safe_min, key_filter=key_filter
        )

    def _filter_keys(self, batch: Batch) -> Batch:
        """Filter the batch using the input_filter.

        Args:
          batch: batch to filter
          batch: Batch:

        Returns:


        """
        if self.input_filter is None:
            return batch

        return {k: v for k, v in batch.items() if self.input_filter(k)}

    def _check_cached_fingerprint(
        self,
        cache_dir: PathLike,
        dataset: Dataset,
        kwargs: Optional[Dict] = None,
        fingerprint_kwargs_exclude: Optional[List[str]] = None,
        debug: bool = True,
    ):
        """This method checks if the cached fingerprint is the same as the current one and save
        the current one. The cached fingerprint is based on the `Pipe.fingerprint()`,
        `Dataset._fingerprint` and `get_fingerprint(kwargs)`, kwargs matching
        `fingerprint_kwargs_exclude` are exlucded.

        Args:
          cache_dir: Path to the cache directory
          dataset: Dataset to process
          kwargs: Additional attributes passed to the pipe
          debug:
          cache_dir: PathLike:
          dataset: Dataset:
          kwargs: Optional[Dict]:  (Default value = None)
          fingerprint_kwargs_exclude: Optional[List[str]]:  (Default value = None)
          debug: bool:  (Default value = True)

        Returns:


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
