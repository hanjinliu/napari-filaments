from __future__ import annotations

from typing import TYPE_CHECKING

import napari

if TYPE_CHECKING:  # pragma: no cover
    from ._widget import FilamentAnalyzer


def start() -> FilamentAnalyzer:
    """Lauch viewer with a FilamentAnalyzer widget docked in it."""
    from ._widget import FilamentAnalyzer

    viewer = napari.Viewer()
    ui = FilamentAnalyzer()
    viewer.window.add_dock_widget(ui)
    return ui
