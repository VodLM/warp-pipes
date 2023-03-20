from timeit import timeit
from warp_pipes.pipes import Sequential, Apply, DropKeys, TokenizerPipe
from transformers import AutoTokenizer
from datasets import load_dataset, set_caching_enabled

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def proc_prefix_lower(entry, keys: list[str]):
   for i, k in entry.items():
      if k in keys:
         entry[i] = "[PREFIX]_" + entry[k].lower()
   return entry

# iterate through a generator until the end
start = timeit()
set_caching_enabled(False)
dataset = load_dataset("squad", split='train', streaming=True)
dataset = dataset.map(lambda x: proc_prefix_lower(x, ['title', 'question'])) # 'answers'
dataset = dataset.remove_columns(["answers"])#(lambda x: x['title'].startswith('Ar'))
dataset = dataset.map(lambda x: tokenizer(x['title']))
# dataset = dataset.map(lambda x: tokenizer(x['context']))
for i, entry in enumerate(dataset):
   ...
end = timeit()
print(end - start)


# warp-pipes example
start = timeit()
set_caching_enabled(False)
dataset = load_dataset("squad", split='train', streaming=False)
plan = Sequential(
   Apply(ops = {'title': lambda x: "[PREFIX]_" + x.lower(), 'question': lambda x: "[PREFIX]_" + x.lower()}, element_wise=True),
   DropKeys(keys = ['answers']),
   TokenizerPipe(tokenizer = tokenizer, key = 'title'),
   # TokenizerPipe(tokenizer = tokenizer, key = 'context')
)
plan(dataset)
end = timeit()
print(end - start)

