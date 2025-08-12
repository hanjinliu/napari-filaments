from himena import MainWindow
from himena.plugins import register_dock_widget_action


@register_dock_widget_action(title="Filament Analysis", singleton=True)
def open_filament_analyzer(ui: MainWindow):
    """Open the filament analysis dock widget."""
    from napari_filaments._himena._widget import FilamentAnalyzer

    return FilamentAnalyzer(ui)
