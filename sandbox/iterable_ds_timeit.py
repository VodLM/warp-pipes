import timeit

setup_code = """
from warp_pipes.pipes import Sequential, Apply, DropKeys, TokenizerPipe
from transformers import AutoTokenizer
from datasets import load_dataset, set_caching_enabled
set_caching_enabled(True)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
"""


# huggingface iterable datasets example.
# seems like num_proc is not implemented for iterable datasets
main_code_iterable_ds = """
def proc_prefix_lower(entry, keys: list[str]):
   for i, k in entry.items():
      if k in keys:
         entry[i] = "[PREFIX]_" + entry[k].lower()
   return entry

dataset = load_dataset("squad", split='train', streaming=True)
dataset = dataset.map(lambda x: proc_prefix_lower(x, ['title', 'question'])) # 'answers'
dataset = dataset.remove_columns(["answers"])#(lambda x: x['title'].startswith('Ar'))
dataset = dataset.map(lambda x: tokenizer(x['title']))
# dataset = dataset.map(lambda x: tokenizer(x['context']))
for i, entry in enumerate(dataset):
   ...
"""

#huggingface dataset example
main_code_ds = """
def proc_prefix_lower(entry, keys: list[str]):
   for i, k in entry.items():
      if k in keys:
         entry[i] = "[PREFIX]_" + entry[k].lower()
   return entry

dataset = load_dataset("squad", split='train', streaming=False)
dataset = dataset.map(lambda x: proc_prefix_lower(x, ['title', 'question']), num_proc=4) # 'answers'
dataset = dataset.remove_columns(["answers"])#(lambda x: x['title'].startswith('Ar'))
dataset = dataset.map(lambda x: tokenizer(x['title']), num_proc=4)
# dataset = dataset.map(lambda x: tokenizer(x['context']))
"""


# warp-pipes example
main_code_wp = """
dataset = load_dataset("squad", split='train', streaming=False)
plan = Sequential(
   Apply(ops = {'title': lambda x: "[PREFIX]_" + x.lower(), 'question': lambda x: "[PREFIX]_" + x.lower()}, element_wise=True),
   DropKeys(keys = ['answers']),
   TokenizerPipe(tokenizer = tokenizer, key = 'title'),
   # TokenizerPipe(tokenizer = tokenizer, key = 'context')
)
plan(dataset)
"""

#print(timeit.timeit(stmt=main_code_iterable_ds,
#          setup=setup_code,
#          number=5))

print(timeit.timeit(stmt=main_code_ds,
          setup=setup_code,
          number=5))


print(timeit.timeit(stmt=main_code_wp,
          setup=setup_code,
          number=5))