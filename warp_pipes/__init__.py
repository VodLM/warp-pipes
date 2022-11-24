__version__ = "0.1.1"

from .core.fingerprintable import Fingerprintable
from .core.pipe import Pipe
from .core import condition
from .pipes import *  # noqa: F403
from warp_pipes.support.datastruct import Eg, Batch
from warp_pipes.support.datasets_utils import HfDataset
from warp_pipes.support.pretty import get_console_separator, pprint_batch
from warp_pipes.support.fingerprint import get_fingerprint

""" from .core.fingerprintable import Fingerprintable
from .core.pipe import Pipe """


# iterate through the modules in the current package
""" package_dir = Path(__file__).resolve().parent
for (_, module_name, _) in iter_modules([package_dir]):

    # import the module and iterate through its attributes
    module = import_module(f"{__name__}.{module_name}")
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)

        if isclass(attribute) and issubclass(attribute, Pipe):
            # Add the class to this package's variables
            globals()[attribute_name] = attribute
 """
