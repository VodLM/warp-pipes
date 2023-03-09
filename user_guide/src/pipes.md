# Pipes
`warp_pipes` is a library that provides a way to process data through pipes and pipelines. The `Pipe` class is a key component of the library, providing an atomic processing unit that can ingest, modify, and return batches of data. The `Pipe` class is designed to be flexible and can handle various types of input data, including:

- List[Eg]
- Batch
- datasets.Dataset
- datasets.DatasetDict

It is an abstract class that defines a single method _call_batch that takes a batch of data, applies the pipe to it, and returns the modified batch. The Pipe class has several properties that can be used to configure the behavior of the pipe. These include:

- `input_filter`: an optional filter that can be applied to the input batch
- `requires_keys`: a list of the names of the keys required by the pipe
- `update`: a boolean flag that determines whether the input batch should be updated with the output of the pipe
- `_allows_update`: a boolean flag that indicates whether the pipe allows setting update=True
- `_allows_input_filter`: a boolean flag that indicates whether the pipe allows setting a custom input_filter
- `_max_num_proc`: the maximum number of parallel processes to use

# Pipelines
The library also includes `Pipelines`, which are containers for pipes that allow multiple processing steps to be combined into a single unit. Overall, the `warp_pipes` library provides a flexible and extensible way to define and apply custom processing steps to data.

Different variants of `Pipelines` are available in the library, including `Parallelize`, `Sequential`, and `Tokenize`. These pipes all inherit the fingerprintable class to extend their capabilities with features such as safe serialization, multiprocessing, and caching.