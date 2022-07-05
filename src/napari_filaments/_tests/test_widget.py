from pathlib import Path

import napari
import numpy as np
from numpy.testing import assert_allclose

from napari_filaments import FilamentAnalyzer

IMAGE_PATH = Path(__file__).parent / "image.tif"
SAVE_PATH = Path(__file__).parent / "result"
DUMMY_PATH = Path(__file__).parent / "test.tif"

rng = np.random.default_rng(1234)


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

    ui.target_filaments.add([[48, 31], [55, 86]], shape_type="path")
    ui.fit_current(ui.parent_viewer.layers[0])

    ui.clip_left()
    ui.clip_right()
    ui.extend_left()
    ui.extend_right()


def test_io(make_napari_viewer):
    ui = _get_dock_widget(make_napari_viewer)
    ui.open_image(IMAGE_PATH)
    img_layer = ui.target_image

    ui.target_filaments.add([[48, 31], [55, 86]], shape_type="path")
    ui.fit_current(img_layer)

    data0 = ui.target_filaments.data

    ui.save_filaments(ui.target_filaments, SAVE_PATH)
    ui.open_filaments(SAVE_PATH)

    data1 = ui.target_filaments.data

    assert data0 is not data1
    assert_allclose(data0, data1, rtol=1e-3, atol=1e-3)


def test_measure(make_napari_viewer):
    ui = _get_dock_widget(make_napari_viewer)
    ui.open_image(IMAGE_PATH)
    img_layer = ui.target_image

    ui.target_filaments.add([[48, 31], [55, 86]], shape_type="path")
    ui.fit_current(img_layer)

    ui.target_filaments.add([[10, 10], [10, 50]], shape_type="path")

    from napari_filaments._spline import Measurement

    ui.measure_properties(img_layer, properties=Measurement.PROPERTIES)
    ui.measure_properties(
        img_layer, properties=Measurement.PROPERTIES, slices=True
    )


def test_adding_multi_channel(make_napari_viewer):
    ui = _get_dock_widget(make_napari_viewer)
    img = rng.normal(size=(5, 3, 100, 100))
    ui._add_image(img, "TCYX", DUMMY_PATH)

    assert ui.target_image is not None
    assert ui.target_filaments is not None
    assert len(ui["target_image"].choices) == 3


def test_selection(make_napari_viewer):
    ui = _get_dock_widget(make_napari_viewer)
    img = rng.normal(size=(5, 100, 100))
    ui._add_image(img, "TYX", DUMMY_PATH)

    assert ui.target_image is not None
    assert ui.filament is None

    ui.target_filaments.add_paths([[0, 10, 10], [0, 10, 50]])
    assert ui["filament"].choices == (0,)
    ui.target_filaments.add_paths([[0, 20, 10], [0, 20, 50]])
    assert ui["filament"].choices == (0, 1)
    assert ui.filament == 1
    ui.target_filaments.add_paths([[2, 30, 10], [2, 30, 50]])
    assert ui["filament"].choices == (0, 1, 2)
    assert ui.filament == 2

    ui.filament = 1
    assert ui.target_filaments.selected_data == {1}
    assert ui.parent_viewer.dims.current_step[0] == 0
    ui.filament = 0
    assert ui.target_filaments.selected_data == {0}
    assert ui.parent_viewer.dims.current_step[0] == 0
    ui.filament = 2
    assert ui.target_filaments.selected_data == {2}
    assert ui.parent_viewer.dims.current_step[0] == 2

    ui["delete_current"].changed()
    assert ui["filament"].choices == (0, 1)
    assert ui.filament == 1
    assert ui.target_filaments.selected_data == {1}
    assert ui.parent_viewer.dims.current_step[0] == 0

    ui.filament = 0
    ui["delete_current"].changed()
    assert ui["filament"].choices == (0,)
    assert ui.target_filaments.selected_data == {0}
    assert ui.filament == 0


def test_target_change(make_napari_viewer):
    ui = _get_dock_widget(make_napari_viewer)
    img = rng.normal(size=(5, 100, 100))
    ui._add_image(img, "TYX", DUMMY_PATH)
    img = rng.poisson(size=(5, 100, 100))
    ui._add_image(img, "TYX", DUMMY_PATH)

    assert ui.target_filaments is ui.parent_viewer.layers[-1]
    assert ui.target_filaments is ui["target_filaments"].choices[1]
    ui.target_filaments.add_paths([[0, 10, 10], [0, 10, 50]])
    ui.target_filaments.add_paths([[1, 10, 10], [1, 10, 50]])
    ui.target_filaments.add_paths([[2, 10, 10], [2, 10, 50]])
    ui.parent_viewer.dims.set_current_step(0, 2)
    assert ui.filament == 2
    assert ui.parent_viewer.dims.current_step[0] == 2

    ui["target_filaments"].value = ui["target_filaments"].choices[0]
    assert ui.filament is None
    assert ui.parent_viewer.dims.current_step[0] == 2
    ui.target_filaments.add_paths([[0, 10, 10], [0, 10, 40]])
    ui.target_filaments.add_paths([[1, 10, 10], [1, 10, 40]])
    assert ui.filament == 1
    ui.parent_viewer.dims.set_current_step(0, 1)

    ui["target_filaments"].value = ui["target_filaments"].choices[1]
    assert ui.filament == 1
    assert ui.parent_viewer.dims.current_step[0] == 1
    ui.filament = 2

    ui["target_filaments"].value = ui["target_filaments"].choices[0]
    assert ui.filament == 1
