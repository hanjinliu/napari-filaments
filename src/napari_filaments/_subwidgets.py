from magicclass import (
    MagicTemplate,
    abstractapi,
    do_not_record,
    field,
    magicclass,
    magicmenu,
    magictoolbar,
    vfield,
)
from magicclass.widgets import Figure, Separator


# fmt: off
@magictoolbar
class Tools(MagicTemplate):
    @magicmenu(icon="ion:layers")
    class Layers(MagicTemplate):
        """Open, save and create layers"""
        open_image = abstractapi()
        open_filaments = abstractapi()
        add_filaments = abstractapi()

        @magicmenu
        class Import(MagicTemplate):
            from_roi = abstractapi()
        sep0 = field(Separator)
        save_filaments = abstractapi()
        sep1 = field(Separator)
        create_total_intensity = abstractapi()
        # def export_roi(self): ...

    @magicmenu(icon="teenyicons:adjust-horizontal-solid")
    class Parameters(MagicTemplate):
        """
        Global parameters of Filament Analyzer.

        Attributes
        ----------
        lattice_width : int
            The width of the image lattice along a filament. Larger value
            increases filament search range but neighbor filaments may
            affect.
        dx : float
            Δx of filament clipping and extension.
        sigma_range : (float, float)
            The range of sigma to be used for fitting to error function.
            This parameter does not affect fitting results. You'll be
            warned if the fitting result was out of this range.
        target_image_filter : bool
            If true, the choice of target image is filtered for each
            filament layer so that only relevant layers will be shown.
        """
        lattice_width = vfield(17, options={"min": 5, "max": 49}, record=False)  # noqa
        dx = vfield(5.0, options={"min": 1, "max": 50.0}, record=False)
        sigma_range = vfield((0.5, 5.0), record=False)
        target_image_filter = vfield(True, record=False)

    @magicmenu(icon="simple-line-icons:graph")
    class Filaments(MagicTemplate):
        """Filaments operations"""
        copy_filament = abstractapi()
        sep0 = field(Separator)
        measure_properties = abstractapi()
        plot_profile = abstractapi()
        plot_curvature = abstractapi()
        kymograph = abstractapi()

    @magicmenu(icon="tabler:dots")
    class Others(MagicTemplate):
        create_macro = abstractapi()
        show_macro = abstractapi()
        send_widget_to_viewer = abstractapi()

@magicclass(widget_type="tabbed")
class Tabs(MagicTemplate):
    @magicclass(layout="horizontal")
    class Spline(MagicTemplate):
        def __post_init__(self):
            self.margins = (2, 2, 2, 2)

        @magicclass(widget_type="groupbox")
        class Left(MagicTemplate):
            extend_left = abstractapi()
            extend_and_fit_left = abstractapi()
            truncate_left = abstractapi()
            truncate_left_at_inflection = abstractapi()

        @magicclass(widget_type="frame")
        class Both(MagicTemplate):
            fit_filament = abstractapi()
            delete_filament = abstractapi()
            @magicclass(layout="horizontal", properties={"margins": (0, 0, 0, 0)})
            class VBox(MagicTemplate):
                undo = abstractapi()
                redo = abstractapi()

            truncate_at_inflections = abstractapi()

        @magicclass(widget_type="groupbox")
        class Right(MagicTemplate):
            extend_right = abstractapi()
            extend_and_fit_right = abstractapi()
            truncate_right = abstractapi()
            truncate_right_at_inflection = abstractapi()

# fmt: on


@magicclass
class Output(MagicTemplate):
    plt = field(Figure)

    def __post_init__(self):
        self._xdata = []
        self._ydata = []
        self.min_height = 200

    @do_not_record
    def view_data(self):
        """View plot data in a table."""
        from ._widget import FilamentAnalyzer

        xlabel = self.plt.ax.get_xlabel() or "x"
        ylabel = self.plt.ax.get_ylabel() or "y"
        if isinstance(self._ydata, list):
            data = {xlabel: self._xdata}
            for i, y in enumerate(self._ydata):
                data[f"{ylabel}-{i}"] = y
        else:
            data = {xlabel: self._xdata, ylabel: self._ydata}
        tstack = self.find_ancestor(FilamentAnalyzer)._tablestack
        tstack.add_table(data, name="Plot data")
        tstack.show()

    def _plot(self, x, y, clear=True, **kwargs):
        if clear:
            self.plt.cla()
        self.plt.plot(x, y, **kwargs)
        self._xdata = x
        if clear:
            self._ydata = y
        else:
            if isinstance(self._ydata, list):
                self._ydata.append(y)
            else:
                self._ydata = [self._ydata, y]

    def _set_labels(self, x: str, y: str):
        self.plt.xlabel(x)
        self.plt.ylabel(y)
