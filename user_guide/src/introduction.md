<div style="margin: 30px auto; background-color: white; border-radius: 50%; width: 200px; height: 200px;"><img src="https://raw.githubusercontent.com/VodLM/warp-pipes/main/assets/header.png" alt="WarpPipes logo" style="width: 200px; padding: 60px 0px;"></div>

# Introduction
Welcome to the WarpPipes library - a data processing module designed to improve the functionality of HuggingFace datasets. 

This book serves as an introduction to the library's features and how to use it effectively. Through examples and comparisons to other solutions, readers will gain a comprehensive overview of the library.

At the core of the `warp_pipes` library is the `Pipe` class, which provides an atomic processing unit capable of ingesting, modifying, and returning batches of data. These pipes can be combined into `Pipelines`, which are containers that allow multiple processing steps to be combined into a single unit. Overall, the `warp_pipes` library offers a flexible and extensible way to define and apply custom processing steps to data.

If you're a HuggingFace user, this Python package provides a framework for building more flexible and advanced processing steps for your dataset.

## Related projects
- TODO: HuggingFace Community
- TODO: Relation to vod-lm
- TODO: Relation to MedChain

## Performance
- TODO: Showcase how parralizing transformations on datasets (e.g., passages and collates) yields increased performance.