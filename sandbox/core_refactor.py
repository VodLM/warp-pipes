from polars import DataFrame
from dataclasses import dataclass
from functools import singledispatch
from datasets import Dataset


# types that support our protocol
# - map

arrow_dataset = ...
dictionary = ...


class Pipe:


    def __init__(self) -> None:
        pass

    def _from_huggingface(self):
        ...

    def _from_polars(self):
        ...
    
    def _process(self, dataset: Dataset):
        # clip the number of workers
        # create fingerprint
        # run the map execution
        dataset.map()

    def _from_datasets():
        ...


# init with type of data
# execute on that data with a range of transformations


dataset: Dataset = ... # huggingface dataset

# dataframe/dataset/lazyframe

plan = Pipe(dataset)
