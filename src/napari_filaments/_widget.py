import re
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypeVar, Union

import numpy as np
from magicclass import (
    MagicTemplate,
    bind_key,
    do_not_record,
    field,
    magicclass,
    magicmenu,
    magictoolbar,
    set_design,
    set_options,
    vfield,
)
from magicclass.types import Bound, Optional
from magicclass.widgets import Figure
from napari.layers import Image

from ._spline import Spline
from ._types import weight

if TYPE_CHECKING:
    from magicclass.fields import MagicValueField
    from magicgui.widgets import ComboBox
    from typing_extensions import ParamSpec

    _P = ParamSpec("_P")
    _R = TypeVar("_R")

ICON_DIR = Path(__file__).parent / "_icon"
ICON_KWARGS = dict(text="", min_width=42, min_height=42)


def batch(f: "Callable[_P, _R]") -> "Callable[_P, _R]":
    """Create a batch processing using `idx` argument."""

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
    target_image: "MagicValueField[ComboBox, Image]" = vfield(Image)
    lattice_width = vfield(17, options={"min": 5, "max": 49}, record=False)
    dx = vfield(5.0, options={"min": 1, "max": 50.0}, record=False)

    def __post_init__(self):
        self._color_default = np.array([0.973, 1.000, 0.412, 1.000])
        self._last_data: np.ndarray = None

    # fmt: off
    @magictoolbar
    class Tools(MagicTemplate):
        @magicmenu
        class Layers(MagicTemplate):
            def open_image(self): ...
            def add_filament_layer(self): ...

    @magicclass(widget_type="tabbed")
    class Tabs(MagicTemplate):
        @magicclass(layout="horizontal")
        class Spline(MagicTemplate):
            def __post_init__(self):
                self.margins = (2, 2, 2, 2)

            @magicclass(widget_type="groupbox")
            class Left(MagicTemplate):
                def extend_left(self): ...
                def extend_and_fit_left(self): ...
                def clip_left(self): ...
                def clip_left_at_inflection(self): ...

            @magicclass(widget_type="frame")
            class Both(MagicTemplate):
                def fit_current(self): ...
                def delete_current(self): ...
                def undo_spline(self): ...
                def clip_at_inflections(self): ...

            @magicclass(widget_type="groupbox")
            class Right(MagicTemplate):
                def extend_right(self): ...
                def extend_and_fit_right(self): ...
                def clip_right(self): ...
                def clip_right_at_inflection(self): ...

        @magicclass(name="Image")
        class Img(MagicTemplate):
            def create_total_intensity(self): ...

        @magicclass(widget_type="scrollable")
        class Measure(MagicTemplate):
            def measure_length(self): ...
            def plot_profile(self): ...
            plt = field(Figure)

    # fmt: on

    def __init__(self):
        self.layer_paths = None

    def _get_idx(self, w=None) -> Union[int, set[int]]:
        if self.layer_paths is None:
            return 0
        sel = self.layer_paths.selected_data
        if len(sel) == 0:
            return self.layer_paths.nshapes - 1
        return sel

    @Tools.Layers.wraps
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
        self.add_filament_layer(self.parent_viewer.layers[-1], path.stem)
        ndim = self.parent_viewer.dims.ndim
        if ndim == len(axis_labels):
            self.parent_viewer.dims.set_axis_label(
                list(range(ndim)), axis_labels
            )

    @Tools.Layers.wraps
    @set_options(name={"text": "Use default name."})
    def add_filament_layer(
        self, target_image: Image, name: Optional[str] = None
    ):
        if name is None:
            name = target_image.name
            if mactched := re.findall(r"\[.*\](.+)", name):
                name = mactched[0]
        self.layer_paths = self.parent_viewer.add_shapes(
            ndim=target_image.ndim,
            edge_color=self._color_default,
            name=f"[F] {name}",
            edge_width=0.5,
        )
        self.layer_paths.mode = "add_path"

        @self.layer_paths.events.set_data.connect
        def _(e):
            self._last_data = None

    def _update_paths(
        self, idx: int, spl: Spline, current_slice: tuple[int, ...] = ()
    ):
        if idx < 0:
            idx += self.layer_paths.nshapes
        sampled = spl.sample(interval=1.0)
        if current_slice:
            sl = np.stack([np.array(current_slice)] * sampled.shape[0], axis=0)
            sampled = np.concatenate([sl, sampled], axis=1)

        hist = self.layer_paths.data[idx]
        self.layer_paths.add(sampled, shape_type="path")
        self.layer_paths.selected_data = {idx}
        self.layer_paths.remove_selected()
        self._last_data = hist

    def _fit_i_2d(self, width, img, coords) -> Spline:
        spl = Spline.fit(coords, degree=1, err=0.0)
        length = spl.length()
        interv = min(8.0, length / 4)
        rough = spl.fit_filament(
            img, width=width, interval=interv, spline_error=0.0
        )
        return rough.fit_filament(img, width=7, spline_error=3e-2)

    @Tabs.Spline.Both.wraps
    @bind_key("T")
    @set_design(**ICON_KWARGS, icon_path=ICON_DIR / "fit.png")
    @batch
    def fit_current(
        self,
        image: Bound[target_image],
        idx: Bound[_get_idx] = -1,
        width: Bound[lattice_width] = 9,
    ):
        self.layer_paths._finish_drawing()
        data: np.ndarray = self.layer_paths.data[idx]
        current_slice, data = _split_slice_and_path(data)
        fit = self._fit_i_2d(width, image.data[current_slice], data)
        self._update_paths(idx, fit, current_slice)

    def _get_slice_and_spline(
        self, idx: int
    ) -> tuple[tuple[int, ...], Spline]:
        data: np.ndarray = self.layer_paths.data[idx]
        current_slice, data = _split_slice_and_path(data)
        spl = Spline.fit(data, err=0.0)
        return current_slice, spl

    @Tabs.Spline.Both.wraps
    @set_design(**ICON_KWARGS, icon_path=ICON_DIR / "undo.png")
    def undo_spline(self):
        if self._last_data is None:
            return
        idx = self.layer_paths.nshapes - 1
        self.layer_paths.add(self._last_data, shape_type="path")
        self.layer_paths.selected_data = {idx}
        self.layer_paths.remove_selected()
        self._last_data = None

    @Tabs.Spline.Left.wraps
    @set_design(**ICON_KWARGS, icon_path=ICON_DIR / "ext_l.png")
    @batch
    def extend_left(self, idx: Bound[_get_idx] = -1, dx: Bound[dx] = 5.0):
        """Extend spline at the starting edge."""
        current_slice, spl = self._get_slice_and_spline(idx)
        out = spl.extend_left(dx)
        self._update_paths(idx, out, current_slice)

    @Tabs.Spline.Right.wraps
    @set_design(**ICON_KWARGS, icon_path=ICON_DIR / "ext_r.png")
    @batch
    def extend_right(self, idx: Bound[_get_idx], dx: Bound[dx] = 5.0):
        """Extend spline at the ending edge."""
        current_slice, spl = self._get_slice_and_spline(idx)
        out = spl.extend_right(dx)
        self._update_paths(idx, out, current_slice)

    @Tabs.Spline.Left.wraps
    @batch
    @set_design(**ICON_KWARGS, icon_path=ICON_DIR / "extfit_l.png")
    def extend_and_fit_left(
        self,
        image: Bound[target_image],
        idx: Bound[_get_idx],
        dx: Bound[dx] = 5.0,
    ):
        """Extend spline and fit to the filament at the starting edge."""
        current_slice, spl = self._get_slice_and_spline(idx)
        fit = spl.extend_filament_left(
            image.data[current_slice], dx, width=11, spline_error=3e-2
        )
        self._update_paths(idx, fit, current_slice)

    @Tabs.Spline.Right.wraps
    @set_design(**ICON_KWARGS, icon_path=ICON_DIR / "extfit_r.png")
    @batch
    def extend_and_fit_right(
        self,
        image: Bound[target_image],
        idx: Bound[_get_idx],
        dx: Bound[dx] = 5.0,
    ):
        """Extend spline and fit to the filament at the ending edge."""
        current_slice, spl = self._get_slice_and_spline(idx)
        fit = spl.extend_filament_right(
            image.data[current_slice], dx, width=11, spline_error=3e-2
        )
        self._update_paths(idx, fit, current_slice)

    @Tabs.Spline.Left.wraps
    @set_design(**ICON_KWARGS, icon_path=ICON_DIR / "clip_l.png")
    @batch
    def clip_left(self, idx: Bound[_get_idx], dx: Bound[dx] = 5.0):
        """Clip spline at the starting edge."""
        current_slice, spl = self._get_slice_and_spline(idx)
        start = dx / spl.length()
        fit = spl.clip(start, 1.0)
        self._update_paths(idx, fit, current_slice)

    @Tabs.Spline.Right.wraps
    @set_design(**ICON_KWARGS, icon_path=ICON_DIR / "clip_r.png")
    @batch
    def clip_right(self, idx: Bound[_get_idx], dx: Bound[dx] = 5.0):
        """Clip spline at the ending edge."""
        current_slice, spl = self._get_slice_and_spline(idx)
        stop = 1.0 - dx / spl.length()
        fit = spl.clip(0.0, stop)
        self._update_paths(idx, fit, current_slice)

    @Tabs.Spline.Left.wraps
    @set_design(**ICON_KWARGS, icon_path=ICON_DIR / "erf_l.png")
    @batch
    def clip_left_at_inflection(
        self,
        image: Bound[target_image],
        idx: Bound[_get_idx],
    ):
        """Clip spline at the inflection point at starting edge."""
        current_slice, spl = self._get_slice_and_spline(idx)
        fit = spl.clip_at_inflection_left(image.data[current_slice])
        self._update_paths(idx, fit, current_slice)

    @Tabs.Spline.Right.wraps
    @set_design(**ICON_KWARGS, icon_path=ICON_DIR / "erf_r.png")
    @batch
    def clip_right_at_inflection(
        self,
        image: Bound[target_image],
        idx: Bound[_get_idx],
    ):
        """Clip spline at the inflection point at ending edge."""
        current_slice, spl = self._get_slice_and_spline(idx)
        fit = spl.clip_at_inflection_right(image.data[current_slice])
        self._update_paths(idx, fit, current_slice)

    @Tabs.Spline.Both.wraps
    @set_design(**ICON_KWARGS, icon_path=ICON_DIR / "erf2.png")
    @batch
    def clip_at_inflections(
        self,
        image: Bound[target_image],
        idx: Bound[_get_idx],
    ):
        """Clip spline at the inflection points at both ends."""
        current_slice, spl = self._get_slice_and_spline(idx)
        out = spl.clip_at_inflections(image.data[current_slice])
        self._update_paths(idx, out, current_slice)

    @Tabs.Measure.wraps
    def measure_length(self, idx: Bound[_get_idx]):
        _, spl = self._get_slice_and_spline(idx)
        print(spl.length())

    @Tabs.Measure.wraps
    def plot_profile(
        self,
        image: Bound[target_image],
        idx: Bound[_get_idx],
    ):
        """Plot intensity profile."""
        current_slice, spl = self._get_slice_and_spline(idx)
        prof = spl.get_profile(image.data[current_slice])

        self.Tabs.Measure.plt.plot(prof)
        self.Tabs.Measure.plt.show()

    @Tabs.Img.wraps
    @set_options(wlayers={"layout": "vertical"})
    def create_total_intensity(self, wlayers: list[tuple[weight, Image]]):
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
    @set_design(**ICON_KWARGS, icon_path=ICON_DIR / "del.png")
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
