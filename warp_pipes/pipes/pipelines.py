import abc
from collections import OrderedDict
from copy import copy
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import T
from typing import Tuple
from typing import Union

from warp_pipes.core.condition import HasPrefix
from warp_pipes.core.condition import Static
from warp_pipes.core.pipe import Pipe
from warp_pipes.pipes.basics import Identity
from warp_pipes.support.datastruct import Batch
from warp_pipes.support.functional import check_equal_arrays
from warp_pipes.support.pretty import pprint_batch
from warp_pipes.support.pretty import repr_batch


class PipeProcessError(Exception):
    """Base class for other exceptions"""

    def __init__(self, pipeline: Pipe, pipe: Pipe, batch: Batch, **kwargs):

        try:
            batch_repr = repr_batch(batch)
        except Exception:
            batch_repr = type(batch)

        keys = _infer_keys(batch)
        kwargs = {k: type(v) for k, v in kwargs.items()}
        msg = (
            f"Exception thrown by pipe: {type(pipe)} in Pipeline {type(pipeline)} with "
            f"batch of type {type(batch)} with keys={keys} "
            f"and kwargs={kwargs}. Batch=\n{batch_repr}"
        )
        super().__init__(msg)


def _call_pipe_and_handle_exception(
    pipe: Pipe, batch: Batch, pipeline: Pipe = None, **kwargs
) -> Batch:
    try:
        return pipe(batch, **kwargs)
    except PipeProcessError as e:
        raise e
    except Exception as e:
        raise PipeProcessError(pipeline, pipe, batch, **kwargs) from e


def _infer_keys(batch):
    if isinstance(batch, dict):
        keys = list(batch.keys())
    elif isinstance(batch, list):
        eg = batch[0]
        keys = _infer_keys(eg)
    else:
        keys = [f"<couldn't infer keys, leaf type={type(batch)}>"]
    return keys


class Pipeline(Pipe):
    """A class that executes other pipes (Sequential, Gate, Parallel, Block)"""

    __metaclass__ = abc.ABCMeta

    def _call_batch(self, batch: T, **kwargs) -> T:
        return self._call_all_types(batch, **kwargs)

    def _call_egs(self, batch: T, **kwargs) -> T:
        return self._call_all_types(batch, **kwargs)

    @abc.abstractmethod
    def _call_all_types(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:
        """
        _call_batch and _call_egs depends on the pipes stored as attributes.
        Therefore a single method can be implemented.
        Only this method should must implemented by the subclasses.
        """
        raise NotImplementedError

    @classmethod
    def instantiate_test(cls, **kwargs) -> None:
        return None


class Sequential(Pipeline):
    """Execute a sequence of pipes."""

    def __init__(self, *pipes: Optional[Union[Callable, Pipe]], **kwargs):
        super(Sequential, self).__init__(**kwargs)
        self.pipes = [pipe for pipe in pipes if pipe is not None]

    def _call_all_types(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:
        """Call the pipes sequentially."""
        for pipe in self.pipes:
            batch = _call_pipe_and_handle_exception(
                pipe, batch, **kwargs, pipeline=self
            )

        return batch

    @classmethod
    def instantiate_test(cls, **kwargs) -> None:
        return cls(Identity(), Identity())


class Parallel(Sequential):
    """Execute pipes in parallel and merge the outputs"""

    def _call_all_types(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:
        """Call the pipes sequentially."""

        outputs = {}
        for pipe in self.pipes:
            pipe_out = _call_pipe_and_handle_exception(
                pipe, copy(batch), **kwargs, pipeline=self
            )

            # check conflict between pipes
            o_keys = set(outputs.keys())
            pipe_o_keys = set(pipe_out.keys())
            intersection = o_keys.intersection(pipe_o_keys)
            for key in intersection:
                msg = (
                    f"There is a conflict between pipes on key={key}\n"
                    f"\n{repr_batch(outputs, 'outputs', rich=False)}"
                    f"\n{repr_batch(pipe_out, 'pipe output', rich=False)}"
                )
                assert check_equal_arrays(outputs[key], pipe_out[key]), msg

            # update output
            outputs.update(**pipe_out)

        return outputs

    @classmethod
    def instantiate_test(cls, **kwargs) -> None:
        return cls(Identity(), Identity())


class Gate(Pipeline):
    """Execute the pipe if the condition is valid, else execute alt."""

    def __init__(
        self,
        condition: Union[bool, Callable],
        pipe: Optional[Pipe],
        alt: Optional[Pipe] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.condition = condition
        if isinstance(condition, bool) and condition is False:
            self.pipe = None
            self.alt = alt
        elif isinstance(condition, bool) and condition is True:
            self.pipe = pipe
            self.alt = None
        else:
            self.pipe = pipe
            self.alt = alt

    def _call_all_types(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:

        switched_on = self.is_switched_on(batch)

        if switched_on:
            if self.pipe is not None:
                return _call_pipe_and_handle_exception(
                    self.pipe, batch, **kwargs, pipeline=self
                )
            else:
                return {}
        else:
            if self.alt is not None:
                return _call_pipe_and_handle_exception(
                    self.alt, batch, **kwargs, pipeline=self
                )
            else:
                return {}

    def is_switched_on(self, batch):
        if isinstance(self.condition, (bool, int)):
            switched_on = self.condition
        else:
            switched_on = self.condition(batch)
        return switched_on

    @classmethod
    def instantiate_test(cls, **kwargs) -> None:
        return cls(Static(True), Identity(), Identity())


class BlockSequential(Pipeline):
    """A sequence of Pipes organized into blocks"""

    def __init__(self, blocks: List[Tuple[str, Pipe]], pprint: bool = False, **kwargs):
        super(BlockSequential, self).__init__(**kwargs)
        blocks = [(k, b) for k, b in blocks if b is not None]
        self.blocks: OrderedDict[str, Pipe] = OrderedDict(blocks)
        self.pprint = pprint

    def _call_all_types(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:
        """Call the pipes sequentially."""
        for name, block in self.blocks.items():
            if self.pprint:
                pprint_batch(batch, f"{name}::input")
            batch = _call_pipe_and_handle_exception(
                block, batch, **kwargs, pipeline=self
            )
            if self.pprint:
                pprint_batch(batch, f"{name}::output")

        return batch

    @classmethod
    def instantiate_test(cls, **kwargs) -> None:
        return cls([("a", Identity()), ("b", Identity())])


class ParallelbyField(Parallel):
    """Run a pipe for each field"""

    def __init__(self, pipes: Dict[str, Pipe], **kwargs):
        super(ParallelbyField, self).__init__(**kwargs)
        self.pipes = {
            field: Sequential(pipe, input_filter=HasPrefix(f"{field}."))
            for field, pipe in pipes.items()
            if pipe is not None
        }

    @classmethod
    def instantiate_test(cls, **kwargs) -> None:
        return cls({"a": Identity(), "b": Identity()})
