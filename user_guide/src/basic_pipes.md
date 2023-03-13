# WarpPipes basic pipes


The following atomic pipe processes are available. Where there are a lot of pipes available for managing the keys, the `Apply` pipe is the most important when it comes to manipulating the values of the dataset.


| Pipe Name | Description |
|------------|------------|
| Identity   | A pipe that passes a batch without modifying it. |
| GetKey     | Returns a batch containing only the target key. |
| FilterKeys | Filter the keys in the batch given the `Condition` object. |
| DropKeys   | Drop the keys in the current batch. |
| AddPrefix  | Append the keys with a prefix. |
| ReplaceInKeys | Remove a pattern `a` with `b` in all keys |
| RenameKeys | Rename a set of keys using a dictionary |
| Apply | Transform the values in a batch using the transformations registered in `ops`|
| ApplyToAll | Transform the values in a batch using the transformations registered in `ops`|
| Lambda     | Apply a lambda function to the batch. |



The following is an example of how to apply a simple Pipe to a dataset. The `warp_pipes` defines a pipeline consisting of different processes. 

Still in progress.


1. Sequential transformations.
    Sep add, tokenize, 
    Create new column by donig something to the existing
    

    DropKey?

2. How to parallize these.




```python
{{#include examples/basics.py}}
```

