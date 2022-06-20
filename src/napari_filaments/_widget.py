from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
from magicclass import bind_key, magicclass, vfield, MagicTemplate, do_not_record
from magicclass.types import Color, Bound
from ._spline import Spline
from napari.layers import Image

if TYPE_CHECKING:
    ...
    
@magicclass
class FilamentAnalyzer(MagicTemplate):
    color_default = vfield(Color, options={"value": "#F8FF69"}, record=False)
    color_fit = vfield(Color, options={"value": "#FF11D7"}, record=False)
    lattice_width = vfield(17, options={"min": 5, "max": 49}, record=False)
    
    def open_image(self, path: Path):
        from tifffile import imread
        img = imread(path)
        l = self.parent_viewer.add_image(img)
        self.add_layer(l)
        
    def add_layer(self, target_image: Image):
        self.layer_paths = self.parent_viewer.add_shapes(
            ndim=target_image.ndim,
            edge_color=np.asarray(self.color_default), 
            name="Filaments of " + target_image.name,
            edge_width=0.5,
        )
        self.layer_paths.mode = "add_path"
        self.layer_image = target_image
    
    def _fit_i(self, width: int, idx: int):
        data: np.ndarray = self.layer_paths.data[idx]
        # TODO: >3-D
        img = self.layer_image.data
        spl = Spline.fit(data, degree=1, err=0.)
        length = spl.length()
        interv = min(length, 8.)
        rough = spl.fit_filament(img, width=width, interval=interv, spline_error=0.)
        fit = rough.fit_filament(img, width=7, spline_error=3e-2)
        
        # update data
        data = self.layer_paths.data
        data[idx] = fit.sample(interval=1.0)
        self.layer_paths.data = data
        
        # update color
        ec = self.layer_paths.edge_color
        ec[idx] = self.color_fit
        self.layer_paths.edge_color = ec
    
    def _fit_i_extended(self, width: int, idx: int, distances: tuple[float, float]):
        data: np.ndarray = self.layer_paths.data[idx]
        img = self.layer_image.data
        spl = Spline.fit(data, err=0.)
        fit = spl.extended_fit(img, width=7, spline_error=3e-2, distances=distances)
        
        # update data
        data = self.layer_paths.data
        data[idx] = fit.sample(interval=1.0)
        self.layer_paths.data = data
        
        # update color
        ec = self.layer_paths.edge_color
        ec[idx] = self.color_fit
        self.layer_paths.edge_color = ec
    
    @bind_key("T")
    def fit_current(self, width: Bound[lattice_width]):
        self._fit_i(width, -1)
    
    def extend_start(self, width: Bound[lattice_width]):
        self._fit_i_extended(width, -1, (5., 0))
        
    def extend_end(self, width: Bound[lattice_width]):
        self._fit_i_extended(width, -1, (0, 5.))

    def fit_all(self, width: Bound[lattice_width]):
        for i in range(self.layer_paths.nshapes):
            self._fit_i(width, i)
    
    @do_not_record
    def create_macro(self):
        self.macro.widget.duplicate().show()
