---
title: Warp Pipes v1
subtitle: Attack Plan for the next iteration of warp pipes
author:
    - name: Benjamin Starostka
    - name: Jens Perregaard
    - name: Andreas Motzfeldt
---

The main purpose of **Warp Pipes** is to distribute work when processing a batch of data.
And the processing units are simply function calls to the `map` method; but wrapped in a container enabling dispatcing and serialization/caching.

We need to reformulate the trajectory of warp-pipes.
This document targets areas of improvements and plan for the next version. 

---

# Improvements
* Define our currated list of processing units (aka. pipes) and their effect.
* Ensure each processing unit is fingerprintable (cache - and serializable).
* Ensure each processing unit can handle the effects of descendings i.e., error or success from another unit.
* Processing units can be planned out as steps; in parallel or sequential.
* Clearer naming conventions to better distinquish types of processing units.
    - `DropKeys` changes structure
    - `Tokenize` mutates the existing values
    - `Collate` combines results of units
* Sequential and Parallel pipelines should be renamed. Parallel is not parallel as in running multiple processes simultaneously. It just returns a dictionary instead of a batch...

> Old deprecated code not contributing to the above mentioned items should be archived/discarded.

# New features
Once the improvements is in place we can consider the future feature set of warp-pipes.

## Pipe expressions
> sandbox

This feature/+refactoring suggests adding data manipulation methods directly to `Pipe` objects, instead of writing separate classes for each type of processing step.

In the following example we showcase how this approach might play out in practice.
```python
# Current: planning multiple pipe operations
batch = ...
tokenizer = ...
plan = Sequential(
    FilterKeys(),
    TokenizerPipe(tokenizer, key='text'),
    AddPrefix(),
    DropKeys(keys=['k1', 'k2', 'k3'])
)
plan(batch)

# Polars inspired way of planning
batch = ...
tokenizer = ...
plan = (
    Sequential(batch)
    .filter()
    .tokenize(tokenizer, key='text')
    .add_prefix()
    .drop(keys = ['k1', 'k2', 'k3'])
)
plan()
```

Adding another `dispatch` to the `__call__` method of `Pipe` for Polars `DataFrame` and `LazyFrame` types is also an option that may be helpful in further investigation.

## Query plan & visualization
The query plan allows us to visualize and optimize pipelines. This allows for debugging of steps before initiating computation. Furthermore, it allows us to optimize the plan similar to how package-managers are doing.

**Simple Ascii representation (hierarchical or diagram)**
```python
plan.describe_plan()
```
TODO: input image

**Graphiz visualization**
```python
plan1.show_graph(optimized=False)
```
TODO: input image

## Optimize plan

Here the optimization plan examines the steps and finds that it is silly to add prefixes for all entries
in the batch, if we are to filter it anyway later in the plan.

```python
tokenizer = ...
batch = ...
plan = (
    Sequential(batch)
    .tokenize(tokenizer, key='text')
    #.add_prefix() !<< this step is moved one step down
    #.filter()
    .filter()
    .add_prefix()
    .drop(keys = ['k1', 'k2', 'k3'])
)

q.lazy().describe_optimized_plan()
```

## LangChain prompt processing
TODO: valuable??