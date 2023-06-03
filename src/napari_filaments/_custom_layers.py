from __future__ import annotations

import numpy as np
from magicgui import register_type
from magicgui.widgets.bases import CategoricalWidget
from napari.layers import Shapes
from napari.utils._magicgui import find_viewer_ancestor
from psygnal import Signal
import macrokit as mk

from ._consts import ROI_ID


class FilamentsLayer(Shapes):
    _type_string = "shapes"

    data_added = Signal(np.ndarray)
    data_removed = Signal(dict[int, np.ndarray])
    draw_finished = Signal(np.ndarray)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_properties = {ROI_ID: 0}

    def add(self, *args, **kwargs):
        next_id = self.nshapes
        props = self.current_properties
        props[ROI_ID] = next_id
        self.current_properties = props
        out = super().add(*args, **kwargs)
        self.data_added.emit()
        return out

    def remove_selected(self):
        selected = list(self.selected_data)
        info = {i: self.data[i].copy() for i in selected}
        out = super().remove_selected()
        self.data_removed.emit(info)
        return out

    def _finish_drawing(self, event=None):
        was_creating = self._is_creating
        out = super()._finish_drawing(event)
        if was_creating:
            # NOTE: Emit here. Before calling super class method, the last data
            # may have a duplicated vertex.
            self.draw_finished.emit(self.data[-1])
        return out


def get_filaments_layer(gui: CategoricalWidget):
    viewer = find_viewer_ancestor(gui)
    if not viewer:
        return []
    return [x for x in viewer.layers if isinstance(x, FilamentsLayer)]


register_type(FilamentsLayer, choices=get_filaments_layer)


@mk.register_type(np.ndarray)
def _format_ndarray(x: np.ndarray):
    if x.ndim != 2 or x.shape[1] != 2:
        return mk.symbol(x)
    return mk.symbol(x.round(2).tolist())
