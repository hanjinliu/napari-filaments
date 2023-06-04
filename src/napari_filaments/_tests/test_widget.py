from pathlib import Path
import pytest

import napari
import numpy as np
from numpy.testing import assert_allclose

from napari_filaments import FilamentAnalyzer, start
from magicclass import get_button
import magicclass.testing as mcls_testing

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


def test_magicclass_stuff(make_napari_viewer):
    ui = _get_dock_widget(make_napari_viewer)
    mcls_testing.check_function_gui_buildable(ui)
    mcls_testing.check_tooltip(ui)


def test_start(make_napari_viewer):
    ui = start(make_napari_viewer())
    ui.parent_viewer.close()


def test_fit(make_napari_viewer):
    ui = _get_dock_widget(make_napari_viewer)
    ui.open_image(IMAGE_PATH)

    ui.target_filaments.add([[48, 31], [55, 86]], shape_type="path")
    ui.fit_filament(ui.parent_viewer.layers[0])

    ui.truncate_left()
    ui.truncate_right()
    ui.extend_left()
    ui.extend_right()


def test_io(make_napari_viewer):
    ui = _get_dock_widget(make_napari_viewer)
    ui.open_image(IMAGE_PATH)
    img_layer = ui.target_image

    ui.target_filaments.add([[48, 31], [55, 86]], shape_type="path")
    ui.fit_filament(img_layer)

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
    ui.fit_filament(img_layer)

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

    get_button(ui.delete_filament).changed()
    assert ui["filament"].choices == (0, 1)
    assert ui.filament == 1
    assert ui.target_filaments.selected_data == {1}
    assert ui.parent_viewer.dims.current_step[0] == 0

    ui.filament = 0
    get_button(ui.delete_filament).changed()
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
    assert ui.filament == 1

    ui["target_filaments"].value = ui["target_filaments"].choices[1]
    assert ui.filament == 1
    assert ui.parent_viewer.dims.current_step[0] == 1
    ui.filament = 2

    ui["target_filaments"].value = ui["target_filaments"].choices[0]
    # 0-th filaments layer has less filaments. Set to the last filament.
    assert ui.filament == 1


def test_add_filament_layer(make_napari_viewer):
    from napari_filaments._custom_layers import FilamentsLayer

    ui = _get_dock_widget(make_napari_viewer)
    img = rng.normal(size=(5, 100, 100))
    ui._add_image(img, "TYX", DUMMY_PATH)
    ui.add_filaments()

    assert type(ui.parent_viewer.layers[-1]) is FilamentsLayer
    assert type(ui.parent_viewer.layers[-2]) is FilamentsLayer
    assert ui.parent_viewer.layers[0].visible


def test_extend_and_fit(make_napari_viewer):
    ui = _get_dock_widget(make_napari_viewer)
    ui.open_image(IMAGE_PATH)

    ui.target_filaments.add([[48, 31], [55, 86]], shape_type="path")
    image_layer = ui.parent_viewer.layers[0]
    ui.fit_filament(image_layer)

    ui.truncate_left()
    ui.truncate_right()
    ui.extend_and_fit_left(image_layer)
    ui.extend_and_fit_right(image_layer)


def test_clip_inflection(make_napari_viewer):
    ui = _get_dock_widget(make_napari_viewer)
    ui.open_image(IMAGE_PATH)

    ui.target_filaments.add([[48, 31], [55, 86]], shape_type="path")
    image_layer = ui.parent_viewer.layers[0]
    ui.fit_filament(image_layer)

    ui.extend_left(dx=20)
    ui.extend_right(dx=20)
    ui.truncate_left_at_inflection(image_layer)
    ui.truncate_right_at_inflection(image_layer)
    assert ui.get_spline(0).length() == pytest.approx(62.7, abs=0.1)
