from pathlib import Path
from typing import TYPE_CHECKING

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
    ...


@magicclass
class FilamentAnalyzer(MagicTemplate):
    color_default = vfield(Color, options={"value": "#F8FF69"}, record=False)
    lattice_width = vfield(17, options={"min": 5, "max": 49}, record=False)

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
    def fit_current(self, width: Bound[lattice_width]):
        self._fit_i(width, -1)

    def extend_left(self):
        data: np.ndarray = self.layer_paths.data[-1]
        spl = Spline.fit(data, err=0.0)
        out = spl.extend_left(5.0)
        self._update_paths(-1, out)

    def extend_right(self):
        data: np.ndarray = self.layer_paths.data[-1]
        spl = Spline.fit(data, err=0.0)
        out = spl.extend_right(5.0)
        self._update_paths(-1, out)

    def extend_and_fit_left(self):
        data: np.ndarray = self.layer_paths.data[-1]
        img = self.layer_image.data
        spl = Spline.fit(data, err=0.0)
        fit = spl.extend_filament_left(img, 5, width=11, spline_error=3e-2)
        self._update_paths(-1, fit)

    def extend_and_fit_right(self):
        data: np.ndarray = self.layer_paths.data[-1]
        img = self.layer_image.data
        spl = Spline.fit(data, err=0.0)
        fit = spl.extend_filament_right(img, 5, width=11, spline_error=3e-2)
        self._update_paths(-1, fit)

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
