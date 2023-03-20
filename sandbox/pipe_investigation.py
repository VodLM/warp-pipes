import polars as pl
from pathlib import Path
from typing import Protocol, Optional, T
from warp_pipes.core.pipe import Pipe
from warp_pipes.core.fingerprintable import Fingerprintable
import warp_pipes.pipes.pipelines as pipeline 
from datasets import load_dataset, Dataset, DatasetDict
import sys



import sys, inspect
def print_classes():
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            print(obj)




dataset = load_dataset("findzebra/corpus", split='train')
?dataset
??dataset
dir(dataset)

# create a polars dataframe instance
polars_dataset = pl.from_arrow(dataset.data.table)
polars_dataset.schema
polars_dataset.describe()

q = (
    polars_dataset.select([
        pl.col(['text', 'title', 'source']),
        pl.col('source').str.contains('wikipedia').suffix('_wiki')
    ])
)
q.lazy().show_graph()
print(q.lazy().describe_plan())
print(q.lazy().describe_optimized_plan())

# filter wikipedia articles focusing on memory with valid retrieval date
def filter_proc(entry):
    print(entry)
    title_lower = str.lower(entry['title'])
    if entry['source'] == 'wikipedia' \
        and title_lower.startswith('memory') \
        and entry['retrieved_date'] != None:
        return False
    return True
result = dataset.filter(filter_proc)
pl.from_arrow(result.data.table)




pipeline.Gate()




class HuggingFacePipe(Pipe):
    def __call__(self, input_data: T, **kwargs) -> T:
        return super().__call__(input_data, **kwargs)


# class Pipe(Protocol):
#     def __call__(self, input_data: T, **kwargs) -> T:
#         ...

#     def configure(self):
#         """
#         Configure the internals of the pipe if needed to preparae for some certain steps.
#         """

#     def process(self):
#         """
#         The pipes internal processing step.
#         """
#         ...

# class BatchPipe(Fingerprintable):
#     ...

# class HuggingFacePipe(Fingerprintable):
#     ...