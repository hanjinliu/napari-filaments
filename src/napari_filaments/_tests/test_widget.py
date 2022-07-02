from pathlib import Path

import napari
from numpy.testing import assert_allclose

from napari_filaments import FilamentAnalyzer

IMAGE_PATH = Path(__file__).parent / "image.tif"
SAVE_PATH = Path(__file__).parent / "result"


def _get_dock_widget(make_napari_viewer) -> FilamentAnalyzer:
    viewer: napari.Viewer = make_napari_viewer()

    ui = FilamentAnalyzer()
    viewer.window.add_dock_widget(ui)

    return ui


def test_widget(make_napari_viewer):
    ui = _get_dock_widget(make_napari_viewer)

    assert ui.parent_viewer is not None


def test_fit(make_napari_viewer):
    ui = _get_dock_widget(make_napari_viewer)
    ui.open_image(IMAGE_PATH)

    s0 = [48, 31]
    s1 = [55, 86]

    ui.layer_paths.add([s0, s1], shape_type="path")
    ui.fit_current(ui.parent_viewer.layers[0])

    ui.clip_left()
    ui.clip_right()
    ui.extend_left()
    ui.extend_right()


def test_io(make_napari_viewer):
    ui = _get_dock_widget(make_napari_viewer)
    ui.open_image(IMAGE_PATH)

    s0 = [48, 31]
    s1 = [55, 86]

    ui.layer_paths.add([s0, s1], shape_type="path")
    ui.fit_current(ui.parent_viewer.layers[0])

    data0 = ui.layer_paths.data

    ui.save_filament_layer(ui.layer_paths, SAVE_PATH)
    ui.open_filament_layer(SAVE_PATH)

    data1 = ui.layer_paths.data

    assert data0 is not data1
    assert_allclose(data0, data1, rtol=1e-3, atol=1e-3)
