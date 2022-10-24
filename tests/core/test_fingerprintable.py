from typing import Optional, Any

from warp_pipes.core.fingerprintable import Fingerprintable
import dill
import tempfile
from pathlib import Path

from warp_pipes.support.json_struct import reduce_json_struct


def get_non_pickleable_object():
    """Return a non-pickleable object."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = Path(tmpdir) / "tmp.pkl"
        non_pickleable = open(tmp_file, "w")

    return non_pickleable


def test_get_non_pickleable_object():
    """Test `get_non_pickleable_object`."""
    non_pickleable = get_non_pickleable_object()
    assert dill.pickles(non_pickleable) is False


class MyObject(Fingerprintable):
    """Define a child class of `Fingerprintable`."""

    global_class_attribute = "global_attribute"
    _no_fingerprint = Fingerprintable._no_fingerprint + ["unsafe_attribute"]

    def __init__(
        self,
        *,
        other_attribute="other_attribute",
        unsafe_attribute: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.other_attribute = other_attribute
        self.unsafe_attribute = unsafe_attribute


class MyPickleableObject(MyObject):
    """Define a pickleable version of `MyObject`."""

    def __getstate__(self):
        """Return the state of the object."""
        attrs = self.__dict__
        attrs.pop("unsafe_attribute", None)
        return attrs

    def __setstate__(self, state):
        """Set the state of the object."""
        state["unsafe_attribute"] = None
        self.__dict__.update(state)


def create_nested_object(Cls=MyObject, non_pickleable: Optional[Any] = None):
    """create an instance of MyObject with a MyObject instance as an attribute
    the child MyObject has a non-pickleable attribute"""
    obj_child = Cls(id="child_id", unsafe_attribute=non_pickleable)
    obj_child.dynamic_attr = "dynamic_attr_value"
    obj_parent = Cls(id="parent_id", other_attribute=obj_child)
    return obj_parent


def test_Fingerprintable_dill_inspect():
    """Test `Fingerprintable.dill_inspect`."""
    non_pickleable = get_non_pickleable_object()

    # create an object with non pickleable attributes
    unsafe_object = create_nested_object(MyObject, non_pickleable=non_pickleable)
    # rich.print(unsafe_object.get_fingerprint(reduce=False))
    # rich.print(unsafe_object.dill_inspect(reduce=False))

    # test that the parent object cannot be pickled
    dill_inspection = unsafe_object.dill_inspect(reduce=False)
    assert reduce_json_struct(dill_inspection, all) is False
    dill_inspection = unsafe_object.dill_inspect(reduce=True)
    assert dill_inspection is False

    # lookup the dill status of each attribute
    dill_inspection = unsafe_object.dill_inspect(reduce=False)
    assert dill_inspection["id"] is True
    dill_inspection_child = dill_inspection["other_attribute"]
    assert dill_inspection_child["other_attribute"] is True
    assert dill_inspection_child["unsafe_attribute"] is False
    assert dill_inspection_child["dynamic_attr"] is True

    # create an safe version
    safe_object = create_nested_object(
        MyPickleableObject, non_pickleable=non_pickleable
    )
    # rich.print(safe_object.get_fingerprint(reduce=False))
    # rich.print(safe_object.dill_inspect(reduce=False))

    # test that the class can be pickled when the unsafe attributes are excluded
    dill_inspection = safe_object.dill_inspect(
        reduce=False, exclude_non_fingerprintable=True
    )
    assert reduce_json_struct(dill_inspection, all) is True
    dill_inspection = safe_object.dill_inspect(
        reduce=True, exclude_non_fingerprintable=True
    )
    assert dill_inspection is True

    # lookup the dill status of each attribute
    dill_inspection = safe_object.dill_inspect(reduce=False)
    assert dill_inspection["id"] is True
    dill_inspection_child = dill_inspection["other_attribute"]
    assert dill_inspection_child["other_attribute"] is True
    assert "unsafe_attribute" not in dill_inspection_child.values()
    assert dill_inspection_child["dynamic_attr"] is True

    # clean-up
    non_pickleable.close()


def test_Fingerprintable_get_fingerprint():
    """Test `Fingerprintable.get_fingerprint()`"""

    # create an instance with an fingerprinted attribute "other_aaa"
    # and a non-fingerprinted attribute "unsafe_aaa"
    obj_aaa_aaa = MyObject(other_attribute="other_aaa", unsafe_attribute="unsafe_aaa")
    fingerprint_aaa_aaa = obj_aaa_aaa.get_fingerprint(reduce=True)

    # create an instance with an fingerprinted attribute "other_aaa"
    # and a non-fingerprinted attribute "unsafe_bbb",
    # the fingerprint must be equal to the first one
    obj_aaa_bbb = MyObject(other_attribute="other_aaa", unsafe_attribute="unsafe_bbb")
    fingerprint_aaa_bbb = obj_aaa_bbb.get_fingerprint(reduce=True)
    assert fingerprint_aaa_aaa == fingerprint_aaa_bbb

    # create an instance with an fingerprinted attribute "other_bbb"
    # and a non-fingerprinted attribute "unsafe_aaa",
    # the fingerprint must be diferent from the first
    obj_bbb_aaa = MyObject(other_attribute="other_bbb", unsafe_attribute="unsafe_aaa")
    fingerprint_bbb_aaa = obj_bbb_aaa.get_fingerprint(reduce=True)
    assert fingerprint_aaa_aaa != fingerprint_bbb_aaa
