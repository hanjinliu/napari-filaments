__version__ = "0.5.0"

from ._spline import Spline
from .core import start

from importlib.metadata import metadata

try:
    __version__ = metadata("napari-filaments")["Version"]
except Exception:
    __version__ = "unknown"

del metadata

__all__ = ["Spline", "FilamentAnalyzer", "start"]
