from __future__ import annotations

from abc import ABCMeta
from copy import deepcopy
from functools import partial
from numbers import Number
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import dill
import rich
from loguru import logger

from warp_pipes.support.fingerprint import get_fingerprint
from warp_pipes.support.json_struct import apply_to_json_struct


def leaf_to_json_struct(v: Any, **kwargs) -> Dict | List:
    """Convert a leaf value into a json structure."""
    if isinstance(v, Fingerprintable):
        return v.to_json_struct(**kwargs)
    elif isinstance(v, list):
        return [leaf_to_json_struct(x, **kwargs) for x in v]
    elif isinstance(v, dict):
        return {k: leaf_to_json_struct(x, **kwargs) for k, x in v.items()}
    else:
        return v


class Fingerprintable(object):
    """A `Fingerprintable`s is an object that can be fingeprinted.
    `Fingerprintable` implements a few method that helps integration with the rest of the framework,
    such as safe serialization (pickle, multiprocessing) and deterministic caching (datasets).

    TODO: refactor with __getstate__ and __setstate__ (avoid hacing to specify `no_fingerprint`)

    Functionalities:
     - Serialization capability can be inspected using `.dill_inspect()`
     - The hash/fingerprint of the object and its attributes can be obtained using `.fingerprint()`
     - The object can be reduced to a json-struct using `.to_json_struct()`
     - The object can be copied using `.copy()`
     - The object can be printed using `.pprint()`

     Class attributes:
        - `no_fingerprint`: list of attributes that should not be included in the fingerprint.
        - `id`: an identifier for the Fingerprintable (Optional).
    """

    __metaclass__ = ABCMeta
    no_fingerprint: Optional[List[str]] = ["id"]

    def __init__(self, *, id: Optional[str] = None, **kwargs):
        """
        Args:
            id: (:obj:`Any`, optional): identifier for the Fingerprintable
        """
        self.id = id

    def dill_inspect(
        self, reduce=True, exclude_non_fingerprintable: bool = False
    ) -> bool | Dict[str, bool]:
        """Inspect whether the object can be serialized using dill.

        Args:
          reduce: (:obj:`bool`, optional): test the whole object or test all
            leaves separately and return the corresponding JSON structure.
          exclude_non_fingerprintable: (:obj:`bool`, optional): exclude the `no_fingerprint`
            attributes is set to True

        Returns:
            :obj:`dict`: if `reduce=False`, return a JSON-like structure of booleans indicating
            whether each leaf can be serialized.
            :obj:`bool`: if `reduce=True`, return a boolean indicating whether the object can
            be serialized.

        """

        def safe_pickles(v: Any, key: str, excluded_keys=List[str]) -> str:
            """apply `dill.pickles`, but ignore some keys."""
            if key in excluded_keys:
                return v
            else:
                try:
                    return dill.pickles(v)
                except Exception:
                    return f"<Failed to pickle leaf `{key}` (type={type(v)})>"

        if reduce:
            return dill.pickles(self)
        else:
            data = self.to_json_struct(
                exclude_non_fingerprintable=exclude_non_fingerprintable
            )
            safe_pickles_ = partial(safe_pickles, excluded_keys=["__name__"])
            return apply_to_json_struct(data, safe_pickles_)

    @staticmethod
    def _fingerprint(x: Any) -> str:
        try:
            return get_fingerprint(x)
        except Exception as ex:
            logger.warning(f"Failed to fingerprint {x}: {ex}")

    @staticmethod
    def safe_fingerprint(x: Any, reduce: bool = True) -> Dict | str:
        if isinstance(x, Fingerprintable):
            return x.get_fingerprint(reduce=reduce)
        else:
            return Fingerprintable._fingerprint(x)

    @property
    def fingerprint(self) -> str:
        """Return a fingerprint of the object.
        All attributes stated in `no_fingerprint` are excluded.
        """
        return self.get_fingerprint(reduce=True)

    def get_fingerprint(
        self,
        reduce=False,
    ) -> str | Dict[str, Any]:
        """Return a fingerprint of the object.
        All attributes stated in `no_fingerprint` are excluded.

        Args:
          reduce (:obj:`bool`, optional): if `True`, return a string fingerprint of the object,
          else return a JSON-like structure of fingerprints.

        Returns:
            :obj:`str`: if `reduce=True`, return a JSON-like structure of fingerprints
        """

        fingerprints = self._get_fingerprint_struct()

        if reduce:
            fingerprints = get_fingerprint(fingerprints)

        return fingerprints

    def _get_fingerprint_struct(self) -> List | Dict:
        """get the fingerprint for each element in the JSON-like representation
        of the object, and exclude all parameters stated in `no_fingerprint`

        """
        data = self.to_json_struct(exclude_non_fingerprintable=True)

        def maybe_get_fingerprint(v: Any, key: str) -> str:
            if key == "__name__":
                return v
            elif isinstance(v, Number):
                return str(v)
            elif isinstance(v, str) and len(v) < 32:
                return v
            else:
                return get_fingerprint(v)

        fingerprints = apply_to_json_struct(data, maybe_get_fingerprint)
        return fingerprints

    def to_json_struct(
        self,
        append_self: bool = False,
        exclude: Optional[List[str]] = None,
        exclude_no_recursive: Optional[List[str]] = None,
        include_only: Optional[List[str]] = None,
        include_class_attributes: bool = False,
        exclude_non_fingerprintable: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Return a dictionary representation of the object.

        Args:
          append_self (:obj:`bool`, optional): if `True`, append the object
            itself to the dictionary with key `__self__`
          exclude (`List[str]`, optional): exclude these attributes from the dictionary (key names)
          exclude_no_recursive: (`List[str]`, optional): exclude these attributes from the
            dictionary (key names) but do not exclude their children
          include_only (`List[str]`, optional): include only these attributes in the dictionary
          include_class_attributes (:obj:`bool`, optional): include the class attributes in the dict
          exclude_non_fingerprintable (`bool`, optional): if `True`, extend the exclude
            list with `no_fingerprint`
          **kwargs:

        Returns:
        """
        kwargs = {
            "append_self": append_self,
            "exclude": exclude,
            "include_only": include_only,
            "include_class_attributes": include_class_attributes,
            "exclude_non_fingerprintable": exclude_non_fingerprintable,
            **kwargs,
        }
        attributes = self._get_attributes(
            include_class_attributes=include_class_attributes
        )

        if exclude is None:
            exclude = []

        if exclude_no_recursive is None:
            exclude_no_recursive = []

        if exclude_non_fingerprintable and self.no_fingerprint is not None:
            exclude_no_recursive.extend(self.no_fingerprint)

        exclude = exclude + exclude_no_recursive

        # output data
        data = {"__name__": type(self).__name__, **attributes}
        if append_self:
            data["__self__"] = self

        # filter data
        data = {k: v for k, v in data.items() if v is not None}
        data = {k: v for k, v in data.items() if k not in exclude}
        if include_only is not None:
            data = {k: v for k, v in data.items() if k in include_only}

        # apply the function to the leaf
        data = {k: leaf_to_json_struct(v, **kwargs) for k, v in data.items()}
        return data

    def _get_attributes(self, include_class_attributes: bool = False) -> Dict:
        """Return a dictionary of attributes of the object, uses __getstate__ if available."""
        if include_class_attributes:
            attributes = type(self).__dict__.copy()
        else:
            attributes = {}
        if hasattr(self, "__getstate__"):
            attributes.update(self.__getstate__())
        else:
            attributes.update(vars(self))

        return attributes

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        try:
            attrs = [f"{k}={v}" for k, v in vars(self).items()]
        except RecursionError as err:
            raise err
        return f"{type(self).__name__}({', '.join(attrs)})"

    def copy(self, **kwargs):
        """Return a copy of the object and override the attributes using kwargs."""
        obj = deepcopy(self)
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj

    def __getstate__(self):
        """Return the state of the object."""
        attrs = self.__dict__
        return attrs

    def __setstate__(self, state):
        """Set the state of the object."""
        self.__dict__.update(state)

    def pprint(self):
        rich.print(self.to_json_struct())
