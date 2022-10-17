import importlib
import inspect
import pytest
import rich
import pickle
import dill
from warp_pipes.core.pipe import Pipe
from warp_pipes.pipes import *
import inspect

all_subclasses = set(Pipe.__subclasses__())
n_distinct = len(all_subclasses)
while True:
    for cls in copy(all_subclasses):
        for subclass in cls.__subclasses__():
            all_subclasses.add(subclass)
    if len(all_subclasses) == n_distinct:
        break
    else:
        n_distinct = len(all_subclasses)

@pytest.mark.parametrize("subclass", all_subclasses)
def test_subclasses_pickle_fingerprint(subclass):
    """Test that all the subclasses of `Pipe` can be pickled,
    unpickled, and have deterministic fingerprint"""
    subclass_instance = subclass.instantiate_test()
    if subclass_instance is None:
        return

    # dill inspect
    if not dill.pickles(subclass_instance):
        raise ValueError(f"{subclass} is not pickleable")

    # pickle / unpickle
    buffer = pickle.dumps(subclass_instance)
    subclass_instance_copy = pickle.loads(buffer)

    # fingerprint
    if subclass.instantiate_test().fingerprint != subclass_instance_copy.fingerprint:
        raise ValueError(f"{subclass} fingerprint is not deterministic.")
