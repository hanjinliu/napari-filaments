from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypeVar, Union

import numpy as np
from magicclass import (
    MagicTemplate,
    bind_key,
    do_not_record,
    magicclass,
    vfield,
)
from magicclass.types import Bound, Color
from napari.layers import Image

from ._spline import Spline

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    _P = ParamSpec("_P")
    _R = TypeVar("_R")


def batch(f: "Callable[_P, _R]") -> "Callable[_P, _R]":
    @wraps(f)
    def _f(self, *args, **kwargs):
        if args:
            idx = args[0]
            _args = args[1:]
        else:
            idx = kwargs["idx"]
            _args = ()
        if isinstance(idx, (set, list, tuple)):
            for i in idx:
                f(self, i, *_args, **kwargs)
            return
        f(self, idx, *_args, **kwargs)
        return

    return _f


@magicclass
class FilamentAnalyzer(MagicTemplate):
    color_default = vfield(Color, options={"value": "#F8FF69"}, record=False)
    lattice_width = vfield(17, options={"min": 5, "max": 49}, record=False)
    dx = vfield(5.0, options={"min": 1, "max": 50.0}, record=False)

    def __init__(self):
        self.layer_image = None
        self.layer_paths = None

    def _get_idx(self, w=None) -> Union[int, set[int]]:
        if self.layer_paths is None:
            return 0
        sel = self.layer_paths.selected_data
        if len(sel) == 0:
            return -1
        return sel

    def open_image(self, path: Path):
        from tifffile import imread

        img = imread(path)
        layer = self.parent_viewer.add_image(img)
        self.add_layer(layer)

    def add_layer(self, target_image: Image):
        self.layer_paths = self.parent_viewer.add_shapes(
            ndim=target_image.ndim,
            edge_color=np.asarray(self.color_default),
            name="Filaments of " + target_image.name,
            edge_width=0.5,
        )
        self.layer_paths.mode = "add_path"
        self.layer_image = target_image

    def _update_paths(self, idx: int, spl: Spline):
        data = self.layer_paths.data
        data[idx] = spl.sample(interval=1.0)
        self.layer_paths.data = data

    def _fit_i(self, width: int, idx: int):
        data: np.ndarray = self.layer_paths.data[idx]
        # TODO: >3-D
        img = self.layer_image.data
        spl = Spline.fit(data, degree=1, err=0.0)
        length = spl.length()
        interv = min(length, 8.0)
        rough = spl.fit_filament(
            img, width=width, interval=interv, spline_error=0.0
        )
        fit = rough.fit_filament(img, width=7, spline_error=3e-2)

        self._update_paths(idx, fit)

    @bind_key("T")
    @batch
    def fit_current(
        self, idx: Bound[_get_idx] = -1, width: Bound[lattice_width] = 9
    ):
        self._fit_i(width, idx)

    @batch
    def extend_left(self, idx: Bound[_get_idx] = -1, dx: Bound[dx] = 5.0):
        data: np.ndarray = self.layer_paths.data[idx]
        spl = Spline.fit(data, err=0.0)
        out = spl.extend_left(dx)
        self._update_paths(idx, out)

    @batch
    def extend_right(self, idx: Bound[_get_idx], dx: Bound[dx] = 5.0):
        data: np.ndarray = self.layer_paths.data[idx]
        spl = Spline.fit(data, err=0.0)
        out = spl.extend_right(dx)
        self._update_paths(idx, out)

    @batch
    def extend_and_fit_left(self, idx: Bound[_get_idx], dx: Bound[dx] = 5.0):
        data: np.ndarray = self.layer_paths.data[idx]
        img = self.layer_image.data
        spl = Spline.fit(data, err=0.0)
        fit = spl.extend_filament_left(img, dx, width=11, spline_error=3e-2)
        self._update_paths(idx, fit)

    @batch
    def extend_and_fit_right(self, idx: Bound[_get_idx], dx: Bound[dx] = 5.0):
        data: np.ndarray = self.layer_paths.data[idx]
        img = self.layer_image.data
        spl = Spline.fit(data, err=0.0)
        fit = spl.extend_filament_right(img, dx, width=11, spline_error=3e-2)
        self._update_paths(idx, fit)

    def plot_profile(self):
        data: np.ndarray = self.layer_paths.data[-1]
        img = self.layer_image.data
        spl = Spline.fit(data, err=0.0)
        prof = spl.get_profile(img)
        import matplotlib.pyplot as plt

        plt.plot(prof)
        plt.show()

    def fit_all(self, width: Bound[lattice_width]):
        for i in range(self.layer_paths.nshapes):
            self._fit_i(width, i)

    @do_not_record
    def create_macro(self):
        self.macro.widget.duplicate().show()
