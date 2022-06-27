from pathlib import Path

import napari

from napari_filaments import FilamentAnalyzer

IMAGE_PATH = Path(__file__).parent / "image.tif"


def test_widget(make_napari_viewer):
    viewer: napari.Viewer = make_napari_viewer()

    ui = FilamentAnalyzer()
    viewer.window.add_dock_widget(ui)

    assert ui.parent_viewer is viewer


def test_fit(make_napari_viewer):
    viewer: napari.Viewer = make_napari_viewer()

    ui = FilamentAnalyzer()
    viewer.window.add_dock_widget(ui)
    ui.open_image(IMAGE_PATH)

    s0 = [48, 31]
    s1 = [55, 86]

    ui.layer_paths.add([s0, s1], shape_type="path")
    ui.fit_current(viewer.layers[0])

    ui.clip_left()
    ui.clip_right()
    ui.extend_left()
    ui.extend_right()
