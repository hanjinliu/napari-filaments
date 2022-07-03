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

    ui.layer_paths.add([[48, 31], [55, 86]], shape_type="path")
    ui.fit_current(ui.parent_viewer.layers[0])

    ui.clip_left()
    ui.clip_right()
    ui.extend_left()
    ui.extend_right()


def test_io(make_napari_viewer):
    ui = _get_dock_widget(make_napari_viewer)
    ui.open_image(IMAGE_PATH)
    img_layer = ui.target_image

    ui.layer_paths.add([[48, 31], [55, 86]], shape_type="path")
    ui.fit_current(img_layer)

    data0 = ui.layer_paths.data

    ui.save_filaments(ui.layer_paths, SAVE_PATH)
    ui.open_filaments(SAVE_PATH)

    data1 = ui.layer_paths.data

    assert data0 is not data1
    assert_allclose(data0, data1, rtol=1e-3, atol=1e-3)


def test_measure(make_napari_viewer):
    ui = _get_dock_widget(make_napari_viewer)
    ui.open_image(IMAGE_PATH)
    img_layer = ui.target_image

    ui.layer_paths.add([[48, 31], [55, 86]], shape_type="path")
    ui.fit_current(img_layer)

    ui.layer_paths.add([[10, 10], [10, 50]], shape_type="path")

    from napari_filaments._spline import Measurement

    ui.measure_properties(img_layer, properties=Measurement.PROPERTIES)
    ui.measure_properties(
        img_layer, properties=Measurement.PROPERTIES, slices=True
    )
