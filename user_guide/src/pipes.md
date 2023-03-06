# Pipes
Pipes are considered atomic processing units, flexible in nature and responsible for ingesting, modifying and transforming input datasets into new batches of data. Different variant of such pipes exist namely: `parallelized`, `sequential`, `tokenizationation` etc.. Creating custom pipes can be done by following the core conventions of the `pipe` base object. 

What's common among them are they inherit `fingerprintable` to extend their capabilities with features like safe serialization, multiprocessing and caching.

## Pipelines
Pipelines are like containers for pipes. They orchestrate the configuration and alignment of pipes to perform multiple data processing steps as a combined unit.