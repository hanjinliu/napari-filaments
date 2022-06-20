from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit
from scipy import ndimage as ndi
from scipy.interpolate import splprep, splev

def gaussian(x, mu: float, sg: float, a: float, b: float):
    """1-D Gaussian."""
    return a * np.exp(-(x-mu)**2/(2*sg**2)) + b

def fit(img: np.ndarray) -> np.ndarray:
    """Filament in y-direction."""
    x = np.arange(img.shape[1])
    centers = np.zeros((img.shape[0]), dtype=np.float64)
    for i, line in enumerate(img):
        argmax = np.argmax(line)
        p0 = [argmax, 2, line[argmax], 0]
        params, cov = curve_fit(gaussian, x, line, p0)
        centers[i] = params[0]
    return centers

def get_shift(img: np.ndarray):
    centers = fit(img)
    c0 = (img.shape[1] - 1) / 2
    return centers - c0

class Spline:
    def __init__(self, tck):
        self.tck = tck
    
    def __call__(self, u, *, der: int = 0) -> np.ndarray:
        spl = splev(u, self.tck[0], der=der)
        out = np.stack(spl, axis=1)
        return out
    
    def sample(self, *, npoints: int | None = None, interval: float | None = None):
        if npoints is not None:
            u = np.linspace(0, 1, npoints)
        elif interval is not None:
            npoints = int(self.length() / interval)
            u = np.linspace(0, 1, npoints)
        else:
            raise ValueError("Either `npoints` or `interval` must be given.")
        return self(u)
    
    @classmethod
    def fit(cls, points: np.ndarray, degree: int = 3, err: float = 1e-2):
        npoints = points.shape[0]
        tck = splprep(np.asarray(points).T, s = npoints * err ** 2, k=degree)
        return cls(tck)
    
    def length(self, n: int = 512) -> float:
        out = self(np.linspace(0, 1, n))
        v = np.diff(out, axis=0)
        return np.sum(np.sqrt(np.sum(v ** 2, axis=1)))

    def get_profile(
        self,
        image: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        length = self.length()
        if length < 2:
            raise ValueError(f"Spline is too short ({length:.3f} pixel).")
        u = np.linspace(0, 1, int(length))
        coords = self(u)
        return ndi.map_coordinates(image, coords.T, **kwargs)

    def fit_filament(
        self,
        image: np.ndarray,
        *,
        width: int = 11,
        longitudinal_sigma: float = 1.0,
        interval: float | None = None,
        spline_error: float = 0.,
        **map_kwargs,
    ) -> Spline:
        length = self.length()
        if length < 2:
            raise ValueError(f"Spline is too short ({length:.3f} pixel).")
        int_length = int(length)
        if interval is None:
            interval = length / int_length
            u = np.linspace(0, 1, int_length)
        else:
            u = np.linspace(0, 1, int(length / interval))
        positions = self(u, der=0)
        directions = self(u, der=1)
        directions = directions / np.sqrt(np.sum(directions**2, axis=1))[:, np.newaxis]
        dw = np.arange(width) - (width - 1) / 2  # such as [-2, -1, 0, 1, 2]
        h_dir = np.hstack([directions[:, 1:2], -directions[:, 0:1]])  # rotate by 90 degree
        
        lattice = positions[:, :, np.newaxis] + h_dir[:, :, np.newaxis] * dw[np.newaxis, np.newaxis]  # N, 2, W
        # ready for ndi.map_coordinates
        lat = np.moveaxis(lattice, 1, 0)
        image_trans = ndi.map_coordinates(image, lat, **map_kwargs)
        sigma = longitudinal_sigma / interval
        if sigma > 0:
            ndi.gaussian_filter1d(image_trans, sigma=sigma, axis=0, output=image_trans)
        shift = get_shift(image_trans)
        shift2d: np.ndarray = h_dir * shift[:, np.newaxis]
        out = positions + shift2d

        return self.fit(out, err=spline_error)

    def extend_edges(self, start: float = 0., end: float = 0.) -> Spline:
        points = self.sample(interval=3)
        vec0, vec1 = self([0, 1], der=1)
        n0 = vec0 / np.sqrt(np.sum(vec0**2))
        n1 = vec1 / np.sqrt(np.sum(vec1**2))
        if start > 0:
            e0 = (points[0] - n0 * start).reshape(1, -1)
        else:
            e0 = np.empty((0, 2))
        if end > 0:
            e1 = (points[-1] + n1 * end).reshape(1, -1)
        else:
            e1 = np.empty((0, 2))
        extended = np.concatenate([e0, points, e1], axis=0)
        return self.fit(extended, err=0.)

    def extended_fit(
        self,
        image: np.ndarray,
        *,
        distances: tuple[float, float] = (5, 5),
        width: int = 11,
        longitudinal_sigma: float = 1.0,
        interval: float | None = None,
        spline_error: float = 0.,
        **map_kwargs,
    ) -> Spline:
        spl = self.extend_edges(*distances)
        return spl.fit_filament(
            image,
            width=width,
            longitudinal_sigma=longitudinal_sigma,
            interval=interval,
            spline_error=spline_error,
            **map_kwargs,
        )