import napari

from napari_filaments import FilamentAnalyzer


def test_widget(make_napari_viewer):
    viewer: napari.Viewer = make_napari_viewer()

    ui = FilamentAnalyzer()
    viewer.window.add_dock_widget(ui)

    assert ui.parent_viewer is viewer
