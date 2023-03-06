# Getting started

## Collates process (aka. map and reduce)
In this first example we go trough the steps involved in defining a parralized pipeline that takes in a batch of document objects, runs the preprocessing in parallel, limits the size of the passages and finally collates the result into appropriate tensors to digest.

These steps of operations are very similar to a map/+reduce operation, just with a few extra map operations.

First we tokenize the batch of data in parallel.
```python
{{#include examples/tokenize_in_parallel.py}}
```

Then we generate passages in accordance to some specified constraints.
```python
{{#include examples/generate_passages.py}}
```

Finally we collate the documents into tensors.
```python
{{#include examples/collate_documents.py}}
```