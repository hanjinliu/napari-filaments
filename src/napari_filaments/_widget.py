import re
import weakref
from typing import TYPE_CHECKING, Iterable, TypeVar, Annotated

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import macrokit as mk
from magicclass import (
    MagicTemplate,
    bind_key,
    do_not_record,
    field,
    magicclass,
    nogui,
    set_design,
    set_options,
    vfield,
)
from magicclass.types import Path, Optional
from magicclass.undo import undo_callback
from magicclass.widgets import Figure

import napari
from napari.layers import Image

from . import _optimizer as _opt, _subwidgets as _sw
from ._spline import Measurement, Spline
from ._table_stack import TableStack
from ._types import weight
from ._custom_layers import FilamentsLayer
from ._consts import ROI_ID, TARGET_IMG_LAYERS, IMAGE_AXES, SOURCE

if TYPE_CHECKING:  # pragma: no cover
    from magicgui.widgets import ComboBox

ICON_DIR = Path(__file__).parent / "_icon"
ICON_KW = dict(text="", min_width=42, min_height=42, max_height=45)
SMALL_ICON_KW = dict(text="", min_width=20, min_height=28, max_height=30)

ROI_FMT = "[{" + ROI_ID + "}]"


@mk.register_type(np.ndarray)
def _format_ndarray(x: np.ndarray):
    if x.ndim != 2:
        raise RuntimeError(
            f"{x.ndim}D arrays are not supposed to be an input."
        )
    return str(x.tolist())


@magicclass(widget_type="scrollable")
class FilamentAnalyzer(MagicTemplate):
    """
    Filament Analyzer widget.

    Attributes
    ----------
    target_filaments : Shapes
        The target Shapes layer.
    target_image : Image
        The target Image layer. Fitting/analysis will be performed on this
        layer.
    filament : int
        The selected filament ID. Operations such as filament fitting,
        extension, clipping will be performed on this filament.
    """

    def _get_available_filament_id(self, w=None) -> "list[int]":
        if self.target_filaments is None:
            return []
        return list(range(self.target_filaments.nshapes))

    _tablestack = field(TableStack, name="_Filament Analyzer Tables")
    target_filaments = vfield(FilamentsLayer, record=False)
    target_image = vfield(Image, record=False)
    filament = vfield(int, record=False).with_choices(
        _get_available_filament_id
    )

    Tabs = _sw.Tabs
    Tools = _sw.Tools
    Output = _sw.Output

    def _validate_image_layer(self, image, args) -> str:
        if isinstance(image, str):
            return image
        elif isinstance(image, Image):
            return image.name
        elif image is None:
            return self.target_filaments.name
        else:
            raise TypeError("`image` must be an image layer or its name.")

    def _validate_filaments_layer(self, filaments, args) -> str:
        if isinstance(filaments, str):
            return filaments
        elif isinstance(filaments, FilamentsLayer):
            return filaments.name
        elif filaments is None:
            return self.target_filaments.name
        else:
            raise TypeError(
                "`filaments` must be a filaments layer or its name."
            )

    _ImageLayer = Annotated[
        Image,
        {"bind": target_image, "validator": _validate_image_layer},
    ]
    _FilamentsLayer = Annotated[
        FilamentsLayer,
        {"bind": target_filaments, "validator": _validate_filaments_layer},
    ]

    def __init__(self):
        self._last_target_filaments = None
        self._color_default = np.array([0.973, 1.000, 0.412, 1.000])
        self._nfilaments = 0
        self.objectName()  # activate napari namespace
        self.macro.options.syntax_highlight = True

    def _get_idx(self, w=None) -> "int | set[int]":
        if self.target_filaments is None:
            return 0
        sel = self.target_filaments.selected_data
        if len(sel) == 0:
            if self.target_filaments.nshapes == 0:
                raise ValueError("No filament is selected.")
            return self.target_filaments.nshapes - 1
        return sel

    @property
    def last_target_filaments(self) -> "FilamentsLayer | None":
        if self._last_target_filaments is None:
            return None
        return self._last_target_filaments()

    @last_target_filaments.setter
    def last_target_filaments(self, val: "FilamentsLayer | None"):
        if val is None:
            self._last_target_filaments = None
            return

        if not isinstance(val, FilamentsLayer):
            raise TypeError(
                f"Cannot set type {type(val)} to `last_target_filaments`."
            )
        self._last_target_filaments = weakref.ref(val)
        return None

    @target_filaments.connect
    def _on_target_filament_change(self):
        # old parameters
        _sl = self.parent_viewer.dims.current_step[:-2]
        _fil = self.filament
        if self.last_target_filaments is not None:
            _mode = self.last_target_filaments.mode
            _toggle_target_images(self.last_target_filaments, False)
        else:
            _mode = "pan_zoom"

        _toggle_target_images(self.target_filaments, True)
        self.last_target_filaments = self.target_filaments
        self.parent_viewer.layers.selection = {self.target_filaments}

        self._filter_image_choices()
        cbox: ComboBox = self["filament"]
        cbox.reset_choices()

        # restore old parameters
        self.target_filaments.mode = _mode
        if _fil in cbox.choices:
            cbox.value = _fil
        else:
            try:
                cbox.value = max(cbox.choices)
            except Exception:
                pass
        self.parent_viewer.dims.set_current_step(np.arange(len(_sl)), _sl)
        self._on_filament_change(self.filament)
        return None

    @filament.connect
    def _on_filament_change(self, idx: "int | None"):
        if idx is None:
            return
        layer = self.target_filaments
        data = layer.data[idx]
        _sl, _ = _split_slice_and_path(data)
        self.parent_viewer.dims.set_current_step(np.arange(len(_sl)), _sl)
        layer.selected_data = {idx}

        props = layer.current_properties
        next_id = layer.nshapes
        props[ROI_ID] = next_id
        layer.current_properties = props

        # update text color
        colors = np.full((layer.nshapes, 4), 1.0)
        colors[idx] = layer.edge_color[idx]
        layer.text.color = colors
        if layer.text.color.encoding_type == "ManualColorEncoding":
            layer.text.color.default = "white"
        return None

    @set_design(text="Open image", location=_sw.Tools.Layers)
    @bind_key("Ctrl+K, Ctrl+O")
    def open_image(self, path: Path.Read["*.tif;*.tiff"]):
        """Open a TIF."""
        path = Path(path)
        from tifffile import TiffFile

        with TiffFile(path) as tif:
            series0 = tif.series[0]
            axes = getattr(series0, "axes", "")
            img: np.ndarray = tif.asarray()
        return self._add_image(img, axes, path)

    @set_design(text="Open filaments", location=_sw.Tools.Layers)
    @bind_key("Ctrl+K, Ctrl+F")
    def open_filaments(self, path: Path.Dir):
        """Open a directory with csv files as a filament layer."""
        path = Path(path)

        all_csv: "list[np.ndarray]" = []
        for p in path.glob("*.csv"):
            df = pd.read_csv(p)
            all_csv.append(df.values)
        self._load_filament_coordinates(all_csv, f"[F] {path.stem}")
        return None

    @set_design(text="Add filaments", location=_sw.Tools.Layers)
    def add_filaments(self):
        images = self.target_filaments.metadata[TARGET_IMG_LAYERS]
        name = self.target_filaments.name.lstrip("[F] ")
        return self._add_filament_layer(images, name)

    @set_design(text="From ImageJ ROI", location=_sw.Tools.Layers.Import)
    def from_roi(
        self,
        path: Path.Read["*.zip;*.roi;;All files (*)"],
        filaments: _FilamentsLayer = None,
    ):
        """Import ImageJ Roi zip file as filaments."""
        from roifile import ROI_TYPE, roiread

        _, filaments = self._check_layers(None, filaments)
        path = Path(path)
        rois = roiread(path)
        if not isinstance(rois, list):
            rois = [rois]

        axes = filaments.metadata[IMAGE_AXES]

        for roi in rois:
            if roi.roitype not in (ROI_TYPE.LINE, ROI_TYPE.POLYLINE):
                raise ValueError(f"ROI type {roi.roitype.name} not supported")
            # load coordinates
            yx: np.ndarray = roi.coordinates()[:, ::-1]
            p = roi.position
            t = roi.t_position if "T" in axes else -1
            z = roi.z_position if "Z" in axes else -1

            d = np.array([x - 1 for x in [p, t, z] if x > 0])
            stacked = np.stack([d] * yx.shape[0], axis=0)
            multi_coords = np.concatenate([stacked, yx], axis=1)
            filaments.add_paths(multi_coords)
        return None

    @set_design(text="Save filaments", location=_sw.Tools.Layers)
    @bind_key("Ctrl+K, Ctrl+S")
    def save_filaments(self, layer: FilamentsLayer, path: Path.Save):
        """Save a Shapes layer as a directory of CSV files."""
        import datetime
        import json
        import magicclass as mcls

        from . import __version__

        path = Path(path)
        path.mkdir(exist_ok=True)
        labels = self.parent_viewer.dims.axis_labels
        roi_id = layer.features[ROI_ID]

        ndigits = max(len(str(layer.nshapes)), 2)
        # save filaments
        for idx in range(layer.nshapes):
            data: np.ndarray = layer.data[idx]
            ndim = data.shape[1]
            df = pd.DataFrame(data, columns=list(labels[-ndim:]))
            df.to_csv(
                path / f"Filament-{roi_id[idx]:0>{ndigits}}.csv",
                index=False,
                float_format="%.3f",
            )

        # save other info
        info = {
            "versions": {
                "napari-filaments": __version__,
                "napari": napari.__version__,
                "magicclass": mcls.__version__,
            },
            "date": datetime.datetime.now().isoformat(sep=" "),
            "images": _get_image_sources(layer),
        }
        with open(path / "info.json", "w") as f:
            json.dump(info, f, indent=2)
        return None

    @set_design(
        **ICON_KW, icon=ICON_DIR / "fit.png", location=_sw.Tabs.Spline.Both
    )
    @bind_key("F1")
    def fit_filament(
        self,
        image: Annotated[Image, {"bind": target_image}] = None,
        filaments: _FilamentsLayer = None,
        idx: Annotated[int, {"bind": _get_idx}] = -1,
        width: Annotated[float, {"bind": Tools.Parameters.lattice_width}] = 9,
    ):
        """Fit current spline to the image."""
        image, filaments = self._check_layers(image, filaments)
        filaments._finish_drawing()
        i = _assert_single_selection(idx)
        data: np.ndarray = filaments.data[i]
        current_slice, data = _split_slice_and_path(data)
        fit = self._fit_i_2d(width, image.data[current_slice], data)
        return self._update_paths(i, fit, filaments, current_slice)

    @set_design(
        **SMALL_ICON_KW,
        icon=ICON_DIR / "undo.png",
        location=_sw.Tabs.Spline.Both.VBox,
    )
    @bind_key("Ctrl+Z")
    @do_not_record(recursive=False)
    def undo(self):
        """Undo the last operation."""
        return self.macro.undo()

    @set_design(
        **SMALL_ICON_KW,
        icon=ICON_DIR / "redo.png",
        location=_sw.Tabs.Spline.Both.VBox,
    )
    @bind_key("Ctrl+Y")
    @do_not_record(recursive=False)
    def redo(self):
        """Redo the last spline fit."""
        return self.macro.redo()

    @set_design(
        **ICON_KW, icon=ICON_DIR / "ext_l.png", location=_sw.Tabs.Spline.Left
    )
    def extend_left(
        self,
        filaments: _FilamentsLayer = None,
        idx: Annotated[int, {"bind": _get_idx}] = -1,
        dx: Annotated[float, {"bind": Tools.Parameters.dx}] = 5.0,
    ):
        """Extend spline at the starting edge."""
        _, filaments = self._check_layers(None, filaments)
        idx = _assert_single_selection(idx)
        current_slice, spl = self._get_slice_and_spline(idx, filaments)
        out = spl.extend_left(dx)
        return self._update_paths(idx, out, filaments, current_slice)

    @set_design(
        **ICON_KW, icon=ICON_DIR / "ext_r.png", location=_sw.Tabs.Spline.Right
    )
    def extend_right(
        self,
        filaments: _FilamentsLayer = None,
        idx: Annotated[int, {"bind": _get_idx}] = -1,
        dx: Annotated[float, {"bind": Tools.Parameters.dx}] = 5.0,
    ):
        """Extend spline at the ending edge."""
        _, filaments = self._check_layers(None, filaments)
        idx = _assert_single_selection(idx)
        current_slice, spl = self._get_slice_and_spline(idx, filaments)
        out = spl.extend_right(dx)
        return self._update_paths(idx, out, filaments, current_slice)

    @set_design(
        **ICON_KW,
        icon=ICON_DIR / "extfit_l.png",
        location=_sw.Tabs.Spline.Left,
    )
    def extend_and_fit_left(
        self,
        image: Annotated[Image, {"bind": target_image}] = None,
        filaments: _FilamentsLayer = None,
        idx: Annotated[int, {"bind": _get_idx}] = -1,
        dx: Annotated[float, {"bind": Tools.Parameters.dx}] = 5.0,
    ):
        """Extend spline and fit to the filament at the starting edge."""
        image, filaments = self._check_layers(image, filaments)
        idx = _assert_single_selection(idx)
        current_slice, spl = self._get_slice_and_spline(idx, filaments)
        fit = spl.extend_filament_left(
            image.data[current_slice], dx, width=11, spline_error=3e-2
        )
        return self._update_paths(idx, fit, filaments, current_slice)

    @set_design(
        **ICON_KW,
        icon=ICON_DIR / "extfit_r.png",
        location=_sw.Tabs.Spline.Right,
    )
    def extend_and_fit_right(
        self,
        image: Annotated[Image, {"bind": target_image}] = None,
        filaments: _FilamentsLayer = None,
        idx: Annotated[int, {"bind": _get_idx}] = -1,
        dx: Annotated[float, {"bind": Tools.Parameters.dx}] = 5.0,
    ):
        """Extend spline and fit to the filament at the ending edge."""
        image, filaments = self._check_layers(image, filaments)
        idx = _assert_single_selection(idx)
        current_slice, spl = self._get_slice_and_spline(idx, filaments)
        fit = spl.extend_filament_right(
            image.data[current_slice], dx, width=11, spline_error=3e-2
        )
        return self._update_paths(idx, fit, filaments, current_slice)

    @set_design(
        **ICON_KW, icon=ICON_DIR / "clip_l.png", location=_sw.Tabs.Spline.Left
    )
    def truncate_left(
        self,
        filaments: _FilamentsLayer = None,
        idx: Annotated[int, {"bind": _get_idx}] = -1,
        dx: Annotated[float, {"bind": Tools.Parameters.dx}] = 5.0,
    ):
        """Truncate spline by constant lenght at the starting edge."""
        _, filaments = self._check_layers(None, filaments)
        idx = _assert_single_selection(idx)
        current_slice, spl = self._get_slice_and_spline(idx, filaments)
        start = dx / spl.length()
        fit = spl.clip(start, 1.0)
        return self._update_paths(idx, fit, filaments, current_slice)

    @set_design(
        **ICON_KW, icon=ICON_DIR / "clip_r.png", location=_sw.Tabs.Spline.Right
    )
    def truncate_right(
        self,
        filaments: _FilamentsLayer = None,
        idx: Annotated[int, {"bind": _get_idx}] = -1,
        dx: Annotated[float, {"bind": Tools.Parameters.dx}] = 5.0,
    ):
        """Truncate spline by constant lenght at the ending edge."""
        _, filaments = self._check_layers(None, filaments)
        idx = _assert_single_selection(idx)
        current_slice, spl = self._get_slice_and_spline(idx, filaments)
        stop = 1.0 - dx / spl.length()
        fit = spl.clip(0.0, stop)
        return self._update_paths(idx, fit, filaments, current_slice)

    @set_design(
        **ICON_KW, icon=ICON_DIR / "erf_l.png", location=_sw.Tabs.Spline.Left
    )
    def truncate_left_at_inflection(
        self,
        image: Annotated[Image, {"bind": target_image}] = None,
        filaments: _FilamentsLayer = None,
        idx: Annotated[int, {"bind": _get_idx}] = -1,
    ):
        """Truncate spline at the inflection point at starting edge."""
        image, filaments = self._check_layers(image, filaments)
        idx = _assert_single_selection(idx)
        current_slice, spl = self._get_slice_and_spline(idx, filaments)
        fit = spl.clip_at_inflection_left(
            image.data[current_slice],
            callback=self._show_fitting_result,
        )
        return self._update_paths(idx, fit, filaments, current_slice)

    @set_design(
        **ICON_KW, icon=ICON_DIR / "erf_r.png", location=_sw.Tabs.Spline.Right
    )
    def truncate_right_at_inflection(
        self,
        image: Annotated[Image, {"bind": target_image}] = None,
        filaments: _FilamentsLayer = None,
        idx: Annotated[int, {"bind": _get_idx}] = -1,
    ):
        """Truncate spline at the inflection point at ending edge."""
        image, filaments = self._check_layers(image, filaments)
        idx = _assert_single_selection(idx)
        current_slice, spl = self._get_slice_and_spline(idx, filaments)
        fit = spl.clip_at_inflection_right(
            image.data[current_slice],
            callback=self._show_fitting_result,
        )
        return self._update_paths(idx, fit, filaments, current_slice)

    @set_design(
        **ICON_KW, icon=ICON_DIR / "erf2.png", location=_sw.Tabs.Spline.Both
    )
    def truncate_at_inflections(
        self,
        image: Annotated[Image, {"bind": target_image}] = None,
        filaments: _FilamentsLayer = None,
        idx: Annotated[int, {"bind": _get_idx}] = -1,
    ):
        """Truncate spline at the inflection points at both ends."""
        image, filaments = self._check_layers(image, filaments)
        indices = _arrange_selection(idx)
        for i in indices:
            current_slice, spl = self._get_slice_and_spline(i, filaments)
            out = spl.clip_at_inflections(
                image.data[current_slice],
                callback=self._show_fitting_result,
            )
            self._update_paths(i, out, filaments, current_slice)
        return None

    @set_design(text="Measure properties", location=_sw.Tools.Filaments)
    def measure_properties(
        self,
        image: Annotated[Image, {"bind": target_image}] = None,
        filaments: _FilamentsLayer = None,
        properties: Annotated[
            list[str],
            {"choices": Measurement.PROPERTIES, "widget_type": "Select"},
        ] = ("length", "mean"),
        slices: Annotated[bool, {"label": "Record slice numbers"}] = False,
    ):
        """Measure properties of all the splines."""

        image, filaments = self._check_layers(image, filaments)
        if slices:
            # Record slice numbers in columns such as "index_T"
            ndim = len(image.data.shape)
            labels = self.parent_viewer.dims.axis_labels[-ndim:-2]
            sl_data = {f"index_{lname}": [] for lname in labels}
        else:
            sl_data = {}
        data = {p: [] for p in properties}

        image_data = image.data
        for idx in range(filaments.nshapes):
            sl, spl = self._get_slice_and_spline(idx, filaments)
            measure = Measurement(spl, image_data[sl])
            for v, s0 in zip(sl_data.values(), sl):
                v.append(s0)
            for k, v in data.items():
                v.append(getattr(measure, k)())

        sl_data.update(data)
        tstack = self._tablestack
        tstack.add_table(sl_data, name=filaments.name)
        tstack.show()

        # NOTE: Updating features are not safe. If user added new filaments
        # after measuring, new filaments will initialized with duplicated
        # features.
        # feat = filaments.features
        # filaments.features = feat.assign(**sl_data)
        # defaults = filaments.current_properties
        # for key in defaults.keys():
        #     if key in Measurement.PROPERTIES:
        #         defaults[key] = [np.nan]
        # filaments.current_properties = defaults

        return pd.DataFrame(sl_data)

    @set_design(text="Copy filament", location=Tools.Filaments)
    def copy_filament(
        self,
        idx: Annotated[int, {"bind": _get_idx}] = -1,
        filaments: _FilamentsLayer = None,
        slices: Annotated[
            Optional[list[int]], {"text": "Copy to current slice"}
        ] = None,
    ):
        """Copy a filament in a new slice."""
        _, filaments = self._check_layers(None, filaments)
        data: np.ndarray = filaments.data[idx]
        current_slice, data = _split_slice_and_path(data)
        if slices is None:
            slices = current_slice
        high_dim = np.repeat([slices], data.shape[0], axis=0)
        filaments.add_paths(np.concatenate([high_dim, data], axis=1))
        return None

    @set_design(text="Plot curvature", location=_sw.Tools.Filaments)
    def plot_curvature(
        self,
        idx: Annotated[int, {"bind": _get_idx}] = -1,
        filaments: _FilamentsLayer = None,
    ):
        """Plot curvature of filament."""
        _, filaments = self._check_layers(None, filaments)
        _, spl = self._get_slice_and_spline(idx, filaments)
        length = spl.length()
        x = np.linspace(0, 1, int(spl.length()))
        cv = spl.curvature(x)
        self.Output._plot(x * length, cv)
        self.Output._set_labels("Position (px)", "Curvature")
        return None

    @set_design(text="Plot profile", location=_sw.Tools.Filaments)
    def plot_profile(
        self,
        idx: Annotated[int, {"bind": _get_idx}] = -1,
        image: Annotated[Image, {"bind": target_image}] = None,
        filaments: _FilamentsLayer = None,
    ):
        """Plot intensity profile using the selected image layer and the filament."""
        image, filaments = self._check_layers(image, filaments)
        x, prof = self.get_profile(idx, image, filaments)
        self.Output._plot(x, prof, color=_cmap_to_color(image))
        self.Output._set_labels("Position (px)", "Intensity")
        return None

    def _get_axes(self, w=None) -> "str | None":
        axes = self.parent_viewer.dims.axis_labels[:-2]
        if len(axes) == 0:
            return None
        else:
            if "T" in axes:
                return "T"
            return axes[0]

    @set_design(text="Show kymograph", location=_sw.Tools.Filaments)
    def kymograph(
        self,
        idx: Annotated[int, {"bind": _get_idx}],
        time_axis: Annotated[str, {"bind": _get_axes}],
        image: Annotated[Image, {"bind": target_image}] = None,
        filaments: _FilamentsLayer = None,
    ):
        """Plot kymograph using the selected image layer and the filament."""
        image, filaments = self._check_layers(image, filaments)
        sl, spl = self._get_slice_and_spline(idx, filaments)
        if time_axis is None:
            if image.ndim < 3:
                raise ValueError("Cannot build a kymograph with a 2D image.")
            t0 = image.metadata[IMAGE_AXES][0]
        if isinstance(time_axis, str):
            t0 = image.metadata[IMAGE_AXES].index(time_axis)
        else:
            raise TypeError(f"Invalid time_axis {time_axis!r}.")
        ntime = image.data.shape[t0]
        profiles: "list[np.ndarray]" = []
        for t in range(ntime):
            t1 = t0 + 1
            sl = sl[:t0] + (t,) + sl[t1:]
            prof = spl.get_profile(image.data[sl])
            profiles.append(prof)
        kymo = np.stack(profiles, axis=0)
        plt = Figure()
        plt.imshow(kymo, cmap=_cmap_to_mpl_cmap(image))
        self.parent_viewer.window.add_dock_widget(plt, name="kymograph")
        return None

    @set_options(wlayers={"layout": "vertical", "label": "weight x layer"})
    @set_design(text="Create total intensity", location=_sw.Tools.Layers)
    def create_total_intensity(self, wlayers: list[tuple[weight, Image]]):
        """Create a total intensity layer from multiple images."""
        weights = [t[0] for t in wlayers]
        imgs = [t[1].data for t in wlayers]
        names = [t[1].name for t in wlayers]
        tot = sum(w * img for w, img in zip(weights, imgs))

        outs = set()
        for name in names:
            matched = re.findall(r"\[.*\] (.+)", name)
            if matched:
                outs.add(matched[0])
        if len(outs) == 1:
            new_name = f"[Total] {outs.pop()}"
        else:
            new_name = f"[Total] {outs.pop()} etc."

        tot_layer = self.parent_viewer.add_image(
            tot, name=new_name, visible=False
        )

        # update target images
        for layer in self.parent_viewer.layers:
            if not isinstance(layer, FilamentsLayer):
                continue
            # if all the input images belongs to the same shapes layer, update
            # the target image list.
            img_layers = _get_connected_target_image_layers(layer)
            target_names = [target.name for target in img_layers]
            if all(img_name in target_names for img_name in names):
                img_layers.append(tot_layer)
        return None

    # TODO: how to save at subpixel resolution?
    # @set_options(path={"mode": "w", "filter": ".zip"})
    # @set_design(text="Export ROI", location=Tools.Layers)
    # def export_roi(self, layer: FilamentsLayer, path: Path):
    #     """Export filament layer as a ImageJ ROI.zip file."""
    #
    #     from roifile import roiwrite, ImagejRoi, ROI_TYPE, ROI_OPTIONS
    #     roilist: List[ImagejRoi] = []
    #     multi_labels = self.parent_viewer.dims.axis_labels[:-2]
    #     roi_id = layer.features[ROI_ID]
    #     for i, data in enumerate(layer.data):
    #         multi, coords = _split_slice_and_path(data)
    #         n = len(multi)
    #         dim_kwargs = {
    #             f"{l.lower()}_position": p + 1
    #             for l, p in zip(multi_labels[-n:], multi)
    #         }
    #         h, w = np.max(coords, axis=0)
    #         edge_kwargs = dict(
    #             left=0,
    #             top=0,
    #             right=int(w) + 2,
    #             bottom=int(h) + 2,
    #             n_coordinates=coords.shape[0],
    #         )
    #         roi = ImagejRoi(
    #             roitype=ROI_TYPE.POLYLINE,
    #             options=ROI_OPTIONS.SUB_PIXEL_RESOLUTION,
    #             # integer_coordinates=coords[:, ::-1].astype(np.uint16) + 1,
    #             subpixel_coordinates=coords[:, ::-1] + 1,
    #             name=f"Filament-{roi_id[i]}",
    #             **dim_kwargs,
    #             **edge_kwargs,
    #         )
    #         roilist.append(roi)
    #     roiwrite(path, roilist)

    @set_design(
        **ICON_KW, icon=ICON_DIR / "del.png", location=_sw.Tabs.Spline.Both
    )
    def delete_filament(
        self,
        idx: Annotated[int, {"bind": _get_idx}] = None,
        filaments: _FilamentsLayer = None,
    ):
        """Delete selected (or the last) path."""
        _, filaments = self._check_layers(None, filaments)
        if idx is None:
            idx = {self.filament}
        if isinstance(idx, int):
            idx = {idx}
        # keep current state for undoing
        data_info = {
            i: (
                filaments.data[i],
                filaments.features.iloc[i : i + 1, :],
                filaments.edge_color[i],
            )
            for i in idx
        }

        # select and remove
        filaments.selected_data = idx
        filaments.remove_selected()
        if len(idx) == 1 and filaments.nshapes > 0:
            self.filament = min(list(idx)[0], len(filaments.data) - 1)
            filaments.selected_data = {self.filament}

        @undo_callback
        def _undo():
            shapes = filaments.data
            cur_feat = filaments.features
            cur_ec = filaments.edge_color
            for i, (d, feat, ec) in sorted(
                data_info.items(), key=lambda x: x[0], reverse=True
            ):
                shapes.insert(i, d)
                np.insert(cur_ec, i, ec, axis=0)
                cur_feat = pd.concat(
                    [cur_feat.iloc[:i], feat, cur_feat.iloc[i:]],
                    axis=1,
                    ignore_index=True,
                )

            filaments.data = shapes
            filaments._relabel()

        return _undo

    @set_design(text="Create macro", location=_sw.Tools.Others)
    @do_not_record
    @bind_key("Ctrl+Shift+M")
    def create_macro(self):
        """Create an executable Python script."""
        import macrokit as mk

        new = self.macro.widget.duplicate()
        v = mk.Expr("getattr", [mk.symbol(self), "parent_viewer"])
        new.textedit.value = self.macro.format(
            [(mk.symbol(self.parent_viewer), v)]
        )
        new.show()
        return None

    @set_design(text="Show macro", location=_sw.Tools.Others)
    @do_not_record
    def show_macro(self):
        """Show the macro widget."""
        self.macro.widget.show()
        return None

    @set_design(text="Send widget to viewer", location=_sw.Tools.Others)
    @do_not_record
    def send_widget_to_viewer(self):
        """Add this widget to the viewer console as identifier 'ui'."""
        return self.parent_viewer.update_console({"ui": self})

    @nogui
    def add_filament_data(
        self, data: np.ndarray, layer: "FilamentsLayer | None" = None
    ):
        """
        Add a new filament data to the layer.

        Parameters
        ----------
        data : np.ndarray
            Coordinates of the filament.
        layer : FilamentsLayer, optional
            Target filaments layer.
        """
        data = np.asarray(data)
        _, layer = self._check_layers(None, layer)
        with layer.data_added.blocked():
            layer.add_paths(data)

        @undo_callback
        def _undo():
            layer.data = layer.data[:-1]

        @_undo.with_redo
        def _undo():
            with layer.data_added.blocked():
                layer.add_paths(data)

        self["filament"].reset_choices()
        return _undo

    @nogui
    @do_not_record
    def get_spline(
        self, idx: int, filaments: "FilamentsLayer | None" = None
    ) -> Spline:
        """Get the idx-th spline object."""
        _, filaments = self._check_layers(None, filaments)
        _, spl = self._get_slice_and_spline(idx, filaments)
        return spl

    @nogui
    @do_not_record
    def get_profile(
        self,
        idx: int = -1,
        image: "Image | None" = None,
        filaments: "FilamentsLayer | None" = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Get intensity profile using the selected image layer and the filament."""
        image, filaments = self._check_layers(image, filaments)
        current_slice, spl = self._get_slice_and_spline(idx, filaments)
        prof = spl.get_profile(image.data[current_slice])
        length = spl.length()
        x = np.linspace(0, 1, int(length), dtype=np.float32) * length
        return x, prof

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

    def _replace_data(
        self, idx: int, new_data: np.ndarray, filaments: FilamentsLayer
    ):
        """Replace the idx-th data to the new one."""
        data = filaments.data
        data[idx] = new_data
        filaments.data = data
        self.filament = idx
        filaments.selected_data = {idx}
        return None

    def _update_paths(
        self,
        idx: int,
        spl: Spline,
        filaments: FilamentsLayer,
        current_slice: tuple[int, ...] = (),
        with_undo: bool = True,
    ):
        """
        Update the filament path shape at `idx`-th index of `filaments` layer to the spline
        `spl`. If the layer is >3D, `current_slice` will be the slice index of the 2D image.
        """
        if idx < 0:
            idx += filaments.nshapes
        if spl.length() > 10000:
            raise ValueError("Spline is too long.")
        sampled = spl.sample(interval=1.0)
        if current_slice:
            sl = np.stack([np.array(current_slice)] * sampled.shape[0], axis=0)
            sampled = np.concatenate([sl, sampled], axis=1)

        old_data = filaments.data[idx]
        self._replace_data(idx, sampled, filaments)

        if with_undo:

            @undo_callback
            def _out():
                self._replace_data(idx, old_data, filaments)

            @_out.with_redo
            def _out():
                self._replace_data(idx, sampled, filaments)
                filaments.selected_data = {}
                filaments.refresh()

            return _out
        return None

    def _fit_i_2d(
        self, width: float, img: np.ndarray, coords: np.ndarray
    ) -> Spline:
        spl = Spline.fit(coords, degree=1, err=0.0)
        length = spl.length()
        interv = min(8.0, length / 4)
        rough = spl.fit_filament(
            img, width=width, interval=interv, spline_error=0.0
        )
        return rough.fit_filament(img, width=7, spline_error=3e-2)

    def _load_filament_coordinates(self, data: "list[np.ndarray]", name: str):
        ndata = len(data)
        layer = FilamentsLayer(
            data,
            edge_color=self._color_default,
            name=name,
            shape_type="path",
            edge_width=0.5,
            properties={ROI_ID: np.arange(ndata, dtype=np.uint32)},
            text=dict(string=ROI_FMT, color="white", size=8),
        )
        self.parent_viewer.add_layer(layer)

        self._set_filament_layer(layer)
        layer.current_properties = {ROI_ID: ndata}
        self.target_filaments.metadata[TARGET_IMG_LAYERS] = list(
            filter(lambda x: isinstance(x, Image), self.parent_viewer.layers)
        )
        return None

    def _filter_image_choices(self):
        if not self.Tools.Parameters.target_image_filter:
            return
        target_image_widget: ComboBox = self["target_image"]
        if target_image_widget.value is None:
            return
        cbox_idx = target_image_widget.choices.index(target_image_widget.value)
        img_layers = _get_connected_target_image_layers(self.target_filaments)
        if len(img_layers) > 0:
            target_image_widget.choices = img_layers
            cbox_idx = min(cbox_idx, len(img_layers) - 1)
            target_image_widget.value = target_image_widget.choices[cbox_idx]
        return None

    def _add_image(self, img: np.ndarray, axes: str, path: Path):
        if "C" in axes:
            ic = axes.find("C")
            nchn = img.shape[ic]
            axis_labels: "tuple[str, ...]" = tuple(c for c in axes if c != "C")
            img_layers: "list[Image]" = self.parent_viewer.add_image(
                img,
                channel_axis=ic,
                name=[f"[C{i}] {path.stem}" for i in range(nchn)],
                metadata={IMAGE_AXES: axis_labels, SOURCE: path},
            )
        else:
            axis_labels = tuple(axes)
            _layer = self.parent_viewer.add_image(
                img,
                name=path.stem,
                metadata={IMAGE_AXES: axis_labels, SOURCE: path},
            )
            img_layers = [_layer]

        self._add_filament_layer(img_layers, path.stem)
        ndim = self.parent_viewer.dims.ndim
        if ndim == len(axis_labels):
            self.parent_viewer.dims.set_axis_label(
                list(range(ndim)), axis_labels
            )
        self._on_target_filament_change()  # initialize
        return None

    def _on_data_added(self):
        self["filament"].reset_choices()
        if self.target_filaments.nshapes > 0:
            self.filament = self.target_filaments.nshapes - 1
            self.target_filaments.selected_data = {}
            self.target_filaments.refresh()

    def _on_data_removed(self):
        self["filament"].reset_choices()
        self._on_filament_change(self.filament)

    def _on_data_draw_finished(self, layer: FilamentsLayer):
        with layer.draw_finished.blocked():
            added_data = np.round(layer.data[-1], 2)
            layer.data = layer.data[:-1]
            self.add_filament_data(added_data, layer=layer)
            if layer.nshapes > 0:
                self._on_filament_change(layer.nshapes - 1)
        self["filament"].reset_choices()

    def _set_filament_layer(self, layer: FilamentsLayer):
        layer.data_added.connect(self._on_data_added)
        layer.data_removed.connect(self._on_data_removed)
        layer.draw_finished.connect(self._on_data_draw_finished)
        layer.mode = "add_path"
        self.target_filaments = layer
        return layer

    def _add_filament_layer(self, images: "list[Image]", name: str):
        """Add a Filaments layer for the target image."""
        # check input images
        ndim: int = _get_unique_value(img.ndim for img in images)
        axes: "tuple[str]" = _get_unique_value(
            img.metadata[IMAGE_AXES] for img in images
        )
        layer = FilamentsLayer(
            ndim=ndim,
            edge_color=self._color_default,
            name=f"[F] {name}",
            metadata={IMAGE_AXES: axes, TARGET_IMG_LAYERS: images},
            edge_width=0.5,
            properties={ROI_ID: 0},
            text=dict(string=ROI_FMT, color="white", size=8),
        )
        self.parent_viewer.add_layer(layer)
        return self._set_filament_layer(layer)

    def _get_slice_and_spline(
        self, idx: int, filaments: FilamentsLayer
    ) -> "tuple[tuple[int, ...], Spline]":
        data: np.ndarray = filaments.data[idx]
        current_slice, data = _split_slice_and_path(data)
        if data.shape[0] < 4:
            data = Spline.fit(data, degree=1, err=0).sample(interval=1.0)
        spl = Spline.fit(data, err=0.0)
        return current_slice, spl

    def _check_layers(
        self, image: "Image | None", filaments: "FilamentsLayer | None"
    ) -> "tuple[Image, FilamentsLayer]":
        if image is None:
            image = self.target_image
        elif isinstance(image, str):
            image = self.parent_viewer.layers[image]
        if filaments is None:
            filaments = self.target_filaments
        elif isinstance(filaments, str):
            filaments = self.parent_viewer.layers[filaments]
        if not isinstance(image, Image):
            raise TypeError(f"Invalid image type: {type(image)}")
        if not isinstance(filaments, FilamentsLayer):
            raise TypeError(f"Invalid filament type: {type(filaments)}")
        return image, filaments


def _split_slice_and_path(
    data: np.ndarray,
) -> "tuple[tuple[int, ...], np.ndarray]":
    if data.shape[1] == 2:
        return (), data
    sl: np.ndarray = np.unique(data[:, :-2], axis=0)
    if sl.shape[0] != 1:
        raise ValueError("Spline is not in 2D")
    return tuple(sl.ravel().astype(np.int64)), data[:, -2:]


def _get_connected_target_image_layers(
    shapes: FilamentsLayer,
) -> "list[Image]":
    """Return all connected target image layers."""
    return shapes.metadata.get(TARGET_IMG_LAYERS, [])


def _toggle_target_images(shapes: FilamentsLayer, visible: bool):
    """Set target images to visible or invisible."""
    img_layers = _get_connected_target_image_layers(shapes)
    for img_layer in img_layers:
        if img_layer.name.startswith("[Total]"):
            continue
        img_layer.visible = visible
    shapes.visible = visible


def _assert_single_selection(idx: "int | set[int]") -> int:
    if isinstance(idx, set):
        if len(idx) != 1:
            raise ValueError("Multiple selection")
        return next(iter(idx))
    return idx


def _arrange_selection(idx: "int | set[int]") -> "list[int]":
    if isinstance(idx, int):
        return [idx]
    else:
        return sorted(list(idx), reverse=True)


def _get_image_sources(shapes: FilamentsLayer) -> "list[str] | None":
    """Extract image sources from a shapes layer."""
    img_layers = _get_connected_target_image_layers(shapes)
    if not img_layers:
        return None
    sources = []
    for img in img_layers:
        source = img.metadata.get(SOURCE) or img.source.path
        if source is not None:
            sources.append(str(source))
    if not sources:
        return None
    return sources


_V = TypeVar("_V")


def _get_unique_value(vals: Iterable[_V]) -> _V:
    s = set(vals)
    if len(s) != 1:
        raise ValueError(f"Not a unique value: {s}")
    return next(iter(s))


def _cmap_to_color(image: Image) -> np.ndarray:
    """Convert colormap to color."""
    cmap = image.colormap
    return cmap.map(1)


def _cmap_to_mpl_cmap(image: Image, name="no-name"):
    """Convert colormap to matplotlib colormap."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = image.colormap
    return LinearSegmentedColormap.from_list("custom", cmap.colors)
