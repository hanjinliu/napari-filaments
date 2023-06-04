from ._spline import Spline
from ._widget import FilamentAnalyzer
from .core import start

from importlib_metadata import metadata

try:
    __version__ = metadata("napari-filaments")["Version"]
except Exception:
    __version__ = "unknown"

del metadata

__all__ = ["Spline", "FilamentAnalyzer", "start"]
