from __future__ import annotations

from copy import copy
from typing import Any
from typing import Dict
from typing import List

import pydantic

from warp_pipes.core.fingerprintable import Fingerprintable
from warp_pipes.support.fingerprint import get_fingerprint


class FingerprintableConfig(pydantic.BaseModel, Fingerprintable):
    """Base class for search engine configuration."""

    _no_fingerprint: List[str] = Fingerprintable._no_fingerprint + [
        "no_index_fingerprint",
        "__private_attribute_values__",
        "__fields_set__",
    ]
    _no_index_fingerprint: List[str] = []

    def get_fingerprint(self, reduce: bool = False) -> str | Dict[str, Any]:
        """Fingerprints the arguments used at query time."""
        fingerprints = copy(self.__dict__)
        for exclude in self._no_fingerprint:
            fingerprints.pop(exclude, None)
        if reduce:
            return get_fingerprint(fingerprints)
        else:
            return fingerprints

    def get_indexing_fingerprint(self, reduce: bool = False) -> str:
        """Fingerprints the arguments used at indexing time."""
        fingerprints = self.get_fingerprint(reduce=False)
        for exclude in self._no_index_fingerprint:
            fingerprints.pop(exclude, None)
        if reduce:
            return get_fingerprint(fingerprints)
        else:
            return fingerprints
