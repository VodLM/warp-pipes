from numbers import Number
from typing import Dict
from typing import List
from typing import Union

import numpy as np
from torch import Tensor

Eg = Dict[str, Union[bool, str, Number, Tensor, List, np.ndarray]]
Batch = Dict[str, Union[Tensor, List, np.ndarray]]
