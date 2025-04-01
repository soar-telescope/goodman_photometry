from importlib.metadata import version
from .astrometry import Astrometry # noqa
from .photometry import Photometry # noqa

__version__ = version("goodman_photometry")
