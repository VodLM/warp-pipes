# WarpPipes
WarpPipes build on a concept called `Pipe`. It is a powerful and versatile class that forms the core of many HuggingFace Dataset operations.

With `Pipe`, you can easily 
- rename and remove columns. 
- apply processing functions to each example in a dataset.
- concatenate datasets.
- apply custom formatting transformations. 

All pipes inherit the `fingerprintable` class, which extends their capabilities with features such as safe serialization, multiprocessing, and caching. This makes the performance of WarpPipes very fast and data transformations seamlessly.

With WarpPipes, users can wrap an arbitrary number of pipes into a single `Pipeline`, providing flexibility and control over their data processing.

To better understand the capability of `Pipe`, let's explore some examples.