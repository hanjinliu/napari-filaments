import re
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypeVar, Union

import numpy as np
from magicclass import (
    MagicTemplate,
    bind_key,
    do_not_record,
    magicclass,
    set_options,
    vfield,
)
from magicclass.types import Bound, Color
from napari.layers import Image

from ._spline import Spline
from ._types import weight

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    _P = ParamSpec("_P")
    _R = TypeVar("_R")

Weightened = tuple[weight, Image]


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
    target_image = vfield(Image)
    lattice_width = vfield(17, options={"min": 5, "max": 49}, record=False)
    dx = vfield(5.0, options={"min": 1, "max": 50.0}, record=False)
    color_default = vfield(Color, options={"value": "#F8FF69"}, record=False)

    @magicclass(widget_type="tabbed")
    class Tabs(MagicTemplate):
        @magicclass(layout="horizontal")
        class Spline(MagicTemplate):
            def __post_init__(self):
                self.margins = (2, 2, 2, 2)

            @magicclass(widget_type="groupbox")
            class Left(MagicTemplate):
                def extend_left(self):
                    ...

                def extend_and_fit_left(self):
                    ...

                def clip_left(self):
                    ...

            @magicclass(widget_type="groupbox")
            class Both(MagicTemplate):
                def fit_current(self):
                    ...

                def fit_all(self):
                    ...

                def delete_current(self):
                    ...

                def clip_at_inflections(self):
                    ...

            @magicclass(widget_type="groupbox")
            class Right(MagicTemplate):
                def extend_right(self):
                    ...

                def extend_and_fit_right(self):
                    ...

                def clip_right(self):
                    ...

        @magicclass(name="Image")
        class Img(MagicTemplate):
            def create_total_intensity(self):
                ...

        @magicclass
        class Measure(MagicTemplate):
            def measure_length(self):
                ...

            def plot_profile(self):
                ...

    def __init__(self):
        self.layer_image = None
        self.layer_paths = None

    def _get_idx(self, w=None) -> Union[int, set[int]]:
        if self.layer_paths is None:
            return 0
        sel = self.layer_paths.selected_data
        if len(sel) == 0:
            return self.layer_paths.nshapes - 1
        return sel

    @property
    def image(self) -> np.ndarray:
        return self.target_image.data

    def open_image(self, path: Path):
        path = Path(path)
        from tifffile import TiffFile

        with TiffFile(path) as tif:
            series0 = tif.series[0]
            axes = getattr(series0, "axes", "")
            img: np.ndarray = tif.asarray()

        if "C" in axes:
            ic = axes.find("C")
            nchn = img.shape[ic]
            self.parent_viewer.add_image(
                img,
                channel_axis=ic,
                name=[f"[C{i}] {path.stem}" for i in range(nchn)],
            )
        else:
            self.parent_viewer.add_image(img, name=path.stem)
        axis_labels = [c for c in axes if c != "C"]
        self.add_layer(self.parent_viewer.layers[-1])
        ndim = self.parent_viewer.dims.ndim
        if ndim == len(axis_labels):
            self.parent_viewer.dims.set_axis_label(
                list(range(ndim)), axis_labels
            )

    def add_layer(self, target_image: Image):
        self.layer_paths = self.parent_viewer.add_shapes(
            ndim=target_image.ndim,
            edge_color=np.asarray(self.color_default),
            name="Filaments of " + target_image.name,
            edge_width=0.5,
        )
        self.layer_paths.mode = "add_path"
        self.layer_image = target_image

    def _update_paths(
        self, idx: int, spl: Spline, current_slice: tuple[int, ...] = ()
    ):
        if idx < 0:
            idx += self.layer_paths.nshapes
        sampled = spl.sample(interval=1.0)
        if current_slice:
            sl = np.stack([np.array(current_slice)] * sampled.shape[0], axis=0)
            sampled = np.concatenate([sl, sampled], axis=1)

        self.layer_paths.add(sampled, shape_type="path")
        self.layer_paths.selected_data = {idx}
        self.layer_paths.remove_selected()

    def _fit_i(self, width: int, idx: int):
        data: np.ndarray = self.layer_paths.data[idx]
        current_slice, data = _split_slice_and_path(data)
        img = self.image[current_slice]

        fit = self._fit_i_2d(width, img, data)
        self._update_paths(idx, fit, current_slice)

    def _fit_i_2d(self, width, img, coords) -> Spline:
        spl = Spline.fit(coords, degree=1, err=0.0)
        length = spl.length()
        interv = min(length, 8.0)
        rough = spl.fit_filament(
            img, width=width, interval=interv, spline_error=0.0
        )
        return rough.fit_filament(img, width=7, spline_error=3e-2)

    @Tabs.Spline.Both.wraps
    @bind_key("T")
    @batch
    def fit_current(
        self, idx: Bound[_get_idx] = -1, width: Bound[lattice_width] = 9
    ):
        self.layer_paths._finish_drawing()
        self._fit_i(width, idx)

    def _get_slice_and_spline(
        self, idx: int
    ) -> tuple[tuple[int, ...], Spline]:
        data: np.ndarray = self.layer_paths.data[idx]
        current_slice, data = _split_slice_and_path(data)
        spl = Spline.fit(data, err=0.0)
        return current_slice, spl

    @Tabs.Spline.Left.wraps
    @batch
    def extend_left(self, idx: Bound[_get_idx] = -1, dx: Bound[dx] = 5.0):
        current_slice, spl = self._get_slice_and_spline(idx)
        out = spl.extend_left(dx)
        self._update_paths(idx, out, current_slice)

    @Tabs.Spline.Right.wraps
    @batch
    def extend_right(self, idx: Bound[_get_idx], dx: Bound[dx] = 5.0):
        current_slice, spl = self._get_slice_and_spline(idx)
        out = spl.extend_right(dx)
        self._update_paths(idx, out, current_slice)

    @Tabs.Spline.Left.wraps
    @batch
    def extend_and_fit_left(self, idx: Bound[_get_idx], dx: Bound[dx] = 5.0):
        current_slice, spl = self._get_slice_and_spline(idx)
        fit = spl.extend_filament_left(
            self.image[current_slice], dx, width=11, spline_error=3e-2
        )
        self._update_paths(idx, fit, current_slice)

    @Tabs.Spline.Right.wraps
    @batch
    def extend_and_fit_right(self, idx: Bound[_get_idx], dx: Bound[dx] = 5.0):
        current_slice, spl = self._get_slice_and_spline(idx)
        fit = spl.extend_filament_right(
            self.image[current_slice], dx, width=11, spline_error=3e-2
        )
        self._update_paths(idx, fit, current_slice)

    @Tabs.Spline.Left.wraps
    @batch
    def clip_left(self, idx: Bound[_get_idx], dx: Bound[dx] = 5.0):
        current_slice, spl = self._get_slice_and_spline(idx)
        start = dx / spl.length()
        fit = spl.clip(start, 1.0)
        self._update_paths(idx, fit, current_slice)

    @Tabs.Spline.Right.wraps
    @batch
    def clip_right(self, idx: Bound[_get_idx], dx: Bound[dx] = 5.0):
        current_slice, spl = self._get_slice_and_spline(idx)
        stop = 1.0 - dx / spl.length()
        fit = spl.clip(0.0, stop)
        self._update_paths(idx, fit, current_slice)

    @Tabs.Spline.Both.wraps
    @batch
    def clip_at_inflections(self, idx: Bound[_get_idx]):
        current_slice, spl = self._get_slice_and_spline(idx)
        out = spl.clip_at_inflections(self.image[current_slice])
        self._update_paths(idx, out, current_slice)

    @Tabs.Measure.wraps
    def measure_length(self, idx: Bound[_get_idx]):
        _, spl = self._get_slice_and_spline(idx)
        print(spl.length())

    @Tabs.Measure.wraps
    def plot_profile(self, idx: Bound[_get_idx]):
        current_slice, spl = self._get_slice_and_spline(idx)
        prof = spl.get_profile(self.image[current_slice])
        import matplotlib.pyplot as plt

        plt.plot(prof)
        plt.show()

    @Tabs.Img.wraps
    @set_options(wlayers={"layout": "vertical"})
    def create_total_intensity(self, wlayers: list[Weightened]):
        weights = [t[0] for t in wlayers]
        imgs = [t[1].data for t in wlayers]
        names = [t[1].name for t in wlayers]
        tot = sum(w * img for w, img in zip(weights, imgs))

        outs: set[str] = set()
        for name in names:
            matched = re.findall(r"\[.*\] (.+)", name)
            if matched:
                outs.add(matched[0])
        if len(outs) == 1:
            new_name = f"[Total] {outs.pop()}"
        else:
            new_name = f"[Total] {outs.pop()} etc."
        self.parent_viewer.add_image(tot, name=new_name)

    @Tabs.Spline.Both.wraps
    def fit_all(self, width: Bound[lattice_width]):
        for i in range(self.layer_paths.nshapes):
            self._fit_i(width, i)

    @Tabs.Spline.Both.wraps
    def delete_current(self, idx: Bound[_get_idx]):
        if isinstance(idx, int):
            idx = {idx}
        self.layer_paths.selected_data = idx
        self.layer_paths.remove_selected()

    @do_not_record
    def create_macro(self):
        self.macro.widget.duplicate().show()


def _split_slice_and_path(
    data: np.ndarray,
) -> tuple[tuple[int, ...], np.ndarray]:
    if data.shape[1] == 2:
        return (), data
    sl: np.ndarray = np.unique(data[:, :-2], axis=0)
    if sl.shape[0] != 1:
        raise ValueError("Spline is not in 2D")
    return tuple(sl.ravel().astype(np.int64)), data[:, -2:]
