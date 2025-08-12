from typing import Annotated
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from magicclass import (
    MagicTemplate,
    bind_key,
    magicclass,
    set_design,
    set_options,
)
from himena import MainWindow, StandardType
from himena.widgets import SubWindow
from himena.types import is_subtype
from himena.standards import model_meta, roi

from napari_filaments import _optimizer as _opt
from napari_filaments._himena import _subwidgets as _sw
from napari_filaments._spline import Spline
from napari_filaments._types import weight

ICON_DIR = Path(__file__).parent.parent / "_icon"
ICON_KW = dict(text="", min_width=42, min_height=42, max_height=45)
SMALL_ICON_KW = dict(text="", min_width=20, min_height=28, max_height=30)


@magicclass(widget_type="scrollable")
class FilamentAnalyzer(MagicTemplate):
    """Filament Analyzer widget."""

    Tabs = _sw.Tabs
    Tools = _sw.Tools
    Output = _sw.Output

    def __init__(self, ui: MainWindow):
        self._himena_ui = ui
        self._last_target_filaments = None
        self.macro.options.syntax_highlight = True

    def _get_image_and_roi(
        self, win: SubWindow
    ) -> tuple[NDArray[np.number], roi.SegmentedLineRoi]:
        model = win.to_model()
        if not isinstance(meta := model.metadata, model_meta.ImageMeta):
            raise ValueError("Current window does not have proper metadata.")
        if not isinstance(
            cur_roi := meta.current_roi, (roi.LineRoi, roi.SegmentedLineRoi)
        ):
            raise ValueError("Current ROI is not a line-type ROI.")
        if (ind := meta.current_indices) is None:
            raise ValueError("Current indices not set.")
        sl = tuple(slice(None) if i is None else i for i in ind)
        if isinstance(cur_roi, roi.LineRoi):
            cur_roi = roi.SegmentedLineRoi(
                name=cur_roi.name,
                xs=np.array([cur_roi.x1, cur_roi.x2]),
                ys=np.array([cur_roi.y1, cur_roi.y2]),
            )
        return np.asarray(model.value[sl]), cur_roi

    def _update_current_roi(
        self,
        win: SubWindow,
        spl: Spline,
        current_roi: roi.Roi2D,
    ) -> None:
        if not is_subtype(win.model_type(), StandardType.IMAGE):
            raise ValueError("Current window is not an image.")
        model = win.to_model()
        if not isinstance(meta := model.metadata, model_meta.ImageMeta):
            raise ValueError("Current window does not have proper metadata.")
        if spl.length() > 10000:
            raise ValueError("Spline is too long.")
        sampled = spl.sample(interval=1.0)
        new_roi = roi.SegmentedLineRoi(
            name=current_roi.name,
            xs=sampled[:, 1],
            ys=sampled[:, 0],
        )
        meta = meta.with_current_roi(new_roi)
        meta.skip_image_rerendering = True
        win.update_model(model.with_metadata(meta))

    def _get_current_window(self) -> SubWindow:
        """Get the current window."""
        cur_win = self._himena_ui.current_window
        if not isinstance(cur_win, SubWindow):
            raise ValueError("No window is active")
        if not is_subtype(cur_win.model_type(), StandardType.IMAGE):
            raise ValueError("Current window is not an image.")
        return cur_win

    @set_design(**ICON_KW, icon=ICON_DIR / "fit.png", location=_sw.Tabs.Spline.Both)
    @bind_key("F1")
    def fit_filament(
        self,
        width: Annotated[float, {"bind": Tools.Parameters.lattice_width}] = 9,
    ):
        """Fit current spline to the image."""
        win = self._get_current_window()
        img, cur_roi = self._get_image_and_roi(win)
        fit = self._fit_i_2d(width, img, np.stack([cur_roi.ys, cur_roi.xs], axis=1))
        return self._update_current_roi(win, fit, cur_roi)

    @set_design(**ICON_KW, icon=ICON_DIR / "ext_l.png", location=_sw.Tabs.Spline.Left)
    def extend_left(
        self,
        dx: Annotated[float, {"bind": Tools.Parameters.dx}] = 5.0,
    ):
        """Extend spline at the starting edge."""
        win = self._get_current_window()
        _, cur_roi = self._get_image_and_roi(win)
        spl = self._get_spline_from_roi(cur_roi)
        out = spl.extend_left(dx)
        return self._update_current_roi(win, out, cur_roi)

    @set_design(**ICON_KW, icon=ICON_DIR / "ext_r.png", location=_sw.Tabs.Spline.Right)
    def extend_right(
        self,
        dx: Annotated[float, {"bind": Tools.Parameters.dx}] = 5.0,
    ):
        """Extend spline at the ending edge."""
        win = self._get_current_window()
        _, cur_roi = self._get_image_and_roi(win)
        spl = self._get_spline_from_roi(cur_roi)
        out = spl.extend_right(dx)
        return self._update_current_roi(win, out, cur_roi)

    @set_design(
        **ICON_KW,
        icon=ICON_DIR / "extfit_l.png",
        location=_sw.Tabs.Spline.Left,
    )
    def extend_and_fit_left(
        self,
        dx: Annotated[float, {"bind": Tools.Parameters.dx}] = 5.0,
    ):
        """Extend spline and fit to the filament at the starting edge."""
        win = self._get_current_window()
        img, cur_roi = self._get_image_and_roi(win)
        spl = self._get_spline_from_roi(cur_roi)
        fit = spl.extend_filament_left(img, dx, width=11, spline_error=3e-2)
        return self._update_current_roi(win, fit, cur_roi)

    @set_design(
        **ICON_KW,
        icon=ICON_DIR / "extfit_r.png",
        location=_sw.Tabs.Spline.Right,
    )
    def extend_and_fit_right(
        self,
        dx: Annotated[float, {"bind": Tools.Parameters.dx}] = 5.0,
    ):
        """Extend spline and fit to the filament at the ending edge."""
        win = self._get_current_window()
        img, cur_roi = self._get_image_and_roi(win)
        spl = self._get_spline_from_roi(cur_roi)
        fit = spl.extend_filament_right(img, dx, width=11, spline_error=3e-2)
        return self._update_current_roi(win, fit, cur_roi)

    @set_design(**ICON_KW, icon=ICON_DIR / "clip_l.png", location=_sw.Tabs.Spline.Left)
    def truncate_left(
        self,
        dx: Annotated[float, {"bind": Tools.Parameters.dx}] = 5.0,
    ):
        """Truncate spline by constant length at the starting edge."""
        win = self._get_current_window()
        _, cur_roi = self._get_image_and_roi(win)
        spl = self._get_spline_from_roi(cur_roi)
        start = dx / spl.length()
        fit = spl.clip(start, 1.0)
        return self._update_current_roi(win, fit, cur_roi)

    @set_design(**ICON_KW, icon=ICON_DIR / "clip_r.png", location=_sw.Tabs.Spline.Right)
    def truncate_right(
        self,
        dx: Annotated[float, {"bind": Tools.Parameters.dx}] = 5.0,
    ):
        """Truncate spline by constant length at the ending edge."""
        win = self._get_current_window()
        _, cur_roi = self._get_image_and_roi(win)
        spl = self._get_spline_from_roi(cur_roi)
        start = 1.0 - dx / spl.length()
        fit = spl.clip(0.0, start)
        return self._update_current_roi(win, fit, cur_roi)

    @set_design(**ICON_KW, icon=ICON_DIR / "erf_l.png", location=_sw.Tabs.Spline.Left)
    def truncate_left_at_inflection(self):
        """Truncate spline at the inflection point at starting edge."""
        win = self._get_current_window()
        img, cur_roi = self._get_image_and_roi(win)
        spl = self._get_spline_from_roi(cur_roi)
        fit = spl.clip_at_inflection_left(img, callback=self._show_fitting_result)
        return self._update_current_roi(win, fit, cur_roi)

    @set_design(**ICON_KW, icon=ICON_DIR / "erf_r.png", location=_sw.Tabs.Spline.Right)
    def truncate_right_at_inflection(self):
        """Truncate spline at the inflection point at ending edge."""
        win = self._get_current_window()
        img, cur_roi = self._get_image_and_roi(win)
        spl = self._get_spline_from_roi(cur_roi)
        fit = spl.clip_at_inflection_right(img, callback=self._show_fitting_result)
        return self._update_current_roi(win, fit, cur_roi)

    @set_design(**ICON_KW, icon=ICON_DIR / "erf2.png", location=_sw.Tabs.Spline.Both)
    def truncate_at_inflections(self):
        """Truncate spline at the inflection points at both ends."""
        win = self._get_current_window()
        img, cur_roi = self._get_image_and_roi(win)
        spl = self._get_spline_from_roi(cur_roi)
        fit = spl.clip_at_inflections(img, callback=self._show_fitting_result)
        return self._update_current_roi(win, fit, cur_roi)

    @set_options(wlayers={"layout": "vertical", "label": "weight x layer"})
    @set_design(text="Create total intensity", location=_sw.Tools.Layers)
    def create_total_intensity(self, wlayers: list[tuple[weight, str]]):
        """Create a total intensity layer from multiple images."""
        # weights = [t[0] for t in wlayers]
        # imgs = [t[1].data for t in wlayers]
        # names = [t[1].name for t in wlayers]
        # tot = sum(w * img for w, img in zip(weights, imgs))

        # outs = set()
        # for name in names:
        #     matched = re.findall(r"\[.*\] (.+)", name)
        #     if matched:
        #         outs.add(matched[0])
        # if len(outs) == 1:
        #     new_name = f"[Total] {outs.pop()}"
        # else:
        #     new_name = f"[Total] {outs.pop()} etc."

        # tot_layer = self.parent_viewer.add_image(
        #     tot, name=new_name, visible=False
        # )

        # # update target images
        # for layer in self.parent_viewer.layers:
        #     if not isinstance(layer, FilamentsLayer):
        #         continue
        #     # if all the input images belongs to the same shapes layer, update
        #     # the target image list.
        #     img_layers = _get_connected_target_image_layers(layer)
        #     target_names = [target.name for target in img_layers]
        #     if all(img_name in target_names for img_name in names):
        #         img_layers.append(tot_layer)
        # return None

    def _show_fitting_result(self, opt: _opt.Optimizer, prof: np.ndarray):
        """Callback function for error function fitting"""
        sg_min, sg_max = self.Tools.Parameters.sigma_range
        if isinstance(opt, (_opt.GaussianOptimizer, _opt.ErfOptimizer)):
            valid = sg_min <= opt.params.sg <= sg_max
        elif isinstance(opt, _opt.TwosideErfOptimizer):
            valid0 = sg_min <= opt.params.sg0 <= sg_max
            valid1 = sg_min <= opt.params.sg1 <= sg_max
            valid = valid0 and valid1
        else:
            raise NotImplementedError
        ndata = prof.size
        xdata = np.arange(ndata)
        ydata = opt.sample(xdata)
        self.Output._plot(xdata, prof, color="gray", alpha=0.7, lw=1)
        self.Output._plot(xdata, ydata, clear=False, color="red", lw=2)
        if not valid:
            self.Output.plt.text(
                0, np.min(ydata), "Sigma out of range.", color="crimson"
            )
        return self.Output._set_labels("Data points", "Intensity")

    def _fit_i_2d(self, width: float, img: np.ndarray, coords: np.ndarray) -> Spline:
        spl = Spline.fit(coords, degree=1, err=0.0)
        length = spl.length()
        interv = min(8.0, length / 4)
        rough = spl.fit_filament(img, width=width, interval=interv, spline_error=0.0)
        return rough.fit_filament(img, width=7, spline_error=3e-2)

    def _get_spline_from_roi(self, cur_roi: roi.SegmentedLineRoi) -> "Spline":
        data: np.ndarray = np.stack([cur_roi.ys, cur_roi.xs], axis=1)
        if data.shape[0] < 4:
            data = Spline.fit(data, degree=1, err=0).sample(interval=1.0)
        spl = Spline.fit(data, err=0.0)
        return spl
