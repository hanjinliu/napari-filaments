from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import splev, splprep
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from typing_extensions import Self

    _TCK = tuple[tuple[np.ndarray, list[np.ndarray], int], np.ndarray]


def gaussian(x, mu: float, sg: float, a: float, b: float):
    """1-D Gaussian."""
    return a * np.exp(-((x - mu) ** 2) / (2 * sg**2)) + b


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
    """A spline representation."""

    def __init__(self, tck: _TCK):
        self._tck = tck

    def __hash__(self) -> int:
        return id(self)

    def __call__(self, u, *, der: int = 0) -> np.ndarray:
        scalar = np.isscalar(u)
        if scalar:
            u = [u]
        spl = splev(u, self._tck[0], der=der)
        out = np.stack(spl, axis=1)
        if scalar:
            out = out[0]
        return out

    def sample(
        self, *, npoints: int | None = None, interval: float | None = None
    ):
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
        tck = splprep(np.asarray(points).T, s=npoints * err**2, k=degree)
        return cls(tck)

    @property
    def nknots(self) -> int:
        return self._tck[0][0].size

    @property
    def degree(self) -> int:
        return self._tck[0][2]

    def section(self, range: tuple[float, float], npoints: int = 12) -> Self:
        s0, s1 = range
        if not ((0 <= s0 <= 1) and (0 <= s1 <= 1)):
            raise ValueError("`range` must be a sub-range of [0, 1].")
        coords = self(np.linspace(s0, s1, npoints))
        return self.fit(coords, degree=self.degree, err=0.0)

    def connect(self, other: Self) -> Self:
        coords0 = self.sample(npoints=self.nknots * 3)
        coords1 = other.sample(npoints=other.nknots * 3)
        connection = (coords0[-1:] + coords1[:1]) / 2
        coords = np.concatenate(
            [coords0[:-1], connection, coords1[1:]], axis=0
        )
        return self.fit(coords, degree=max(self.degree, other.degree), err=0.0)

    @lru_cache
    def length(self, n: int = 512) -> float:
        out = self(np.linspace(0, 1, n))
        v = np.diff(out, axis=0)
        return np.sum(np.sqrt(np.sum(v**2, axis=1)))

    def get_profile(
        self,
        image: np.ndarray,
        **map_kwargs,
    ) -> np.ndarray:
        length = self.length()
        if length < 2:
            raise ValueError(f"Spline is too short ({length:.3f} pixel).")
        u = np.linspace(0, 1, int(length))
        coords = self(u)
        return ndi.map_coordinates(image, coords.T, **map_kwargs)

    def measure(
        self,
        image: np.ndarray,
        methods: Sequence[Callable[[np.ndarray], Any]] = (np.mean, np.std),
        **map_kwargs,
    ) -> list:
        prof = self.get_profile(image, **map_kwargs)
        results = []
        for method in methods:
            results.append(method(prof))
        return results

    def fit_filament(
        self,
        image: np.ndarray,
        *,
        width: int = 11,
        degree: int = 3,
        longitudinal_sigma: float = 1.0,
        interval: float | None = None,
        spline_error: float = 0.0,
        **map_kwargs,
    ) -> Self:
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
        directions = (
            directions
            / np.sqrt(np.sum(directions**2, axis=1))[:, np.newaxis]
        )
        dw = np.arange(width) - (width - 1) / 2  # such as [-2, -1, 0, 1, 2]
        h_dir = np.hstack(
            [directions[:, 1:2], -directions[:, 0:1]]
        )  # rotate by 90 degree

        lattice = (
            positions[:, :, np.newaxis]
            + h_dir[:, :, np.newaxis] * dw[np.newaxis, np.newaxis]
        )  # N, 2, W
        # ready for ndi.map_coordinates
        lat = np.moveaxis(lattice, 1, 0)
        image_trans = ndi.map_coordinates(image, lat, **map_kwargs)
        sigma = longitudinal_sigma / interval
        if sigma > 0:
            ndi.gaussian_filter1d(
                image_trans, sigma=sigma, axis=0, output=image_trans
            )
        shift = get_shift(image_trans)
        shift2d: np.ndarray = h_dir * shift[:, np.newaxis]
        out = positions + shift2d

        return self.fit(out, degree=degree, err=spline_error)

    def fit_filament_left(
        self,
        image: np.ndarray,
        border: float,
        *,
        width: int = 11,
        longitudinal_sigma: float = 1.0,
        interval: float | None = None,
        spline_error: float = 0.0,
        **map_kwargs,
    ) -> Self:
        if border < 0 or 1 < border:
            raise ValueError("`border` must be in range of [0, 1].")
        spl_left = self.section((0.0, border))
        spl_right = self.section((border, 1.0))
        spl_fit = spl_left.fit_filament(
            image,
            width=width,
            degree=self.degree,
            longitudinal_sigma=longitudinal_sigma,
            interval=interval,
            spline_error=spline_error,
            **map_kwargs,
        )
        return spl_fit.connect(spl_right)

    def extend_left(
        self,
        length: float,
        *,
        interval: float = 1.0,
    ) -> Self:
        points = self.sample(npoints=self.nknots * 3)
        total_length = self.length()
        vec = self(interval / total_length, der=1)
        norm = vec / np.sqrt(np.sum(vec**2))
        n_ext = int(length / interval)

        edge_points = (
            points[0][np.newaxis]
            - norm[np.newaxis]
            * np.linspace(length / n_ext, length, n_ext)[::-1, np.newaxis]
        )
        extended = np.concatenate([edge_points, points], axis=0)
        return self.fit(extended, degree=self.degree, err=0.0)

    def extend_right(
        self,
        length: float,
        *,
        interval: float = 1.0,
    ) -> Self:
        points = self.sample(npoints=self.nknots * 3)
        total_length = self.length()
        vec = self(1.0 - interval / total_length, der=1)
        norm = vec / np.sqrt(np.sum(vec**2))
        n_ext = int(length / interval)

        edge_points = (
            points[-1][np.newaxis]
            + norm[np.newaxis]
            * np.linspace(length / n_ext, length, n_ext)[:, np.newaxis]
        )
        extended = np.concatenate([points, edge_points], axis=0)
        return self.fit(extended, degree=self.degree, err=0.0)

    def extend_filament_left(
        self,
        image: np.ndarray,
        length: float,
        *,
        width: int = 11,
        longitudinal_sigma: float = 1.0,
        interval: float = 1.0,
        spline_error: float = 0.0,
        **map_kwargs,
    ) -> Self:
        total_length = self.length()

        spl = self.extend_left(length, interval=interval)
        border = (length + 2 * interval) / (total_length + length)
        return spl.fit_filament_left(
            image,
            border,
            width=width,
            longitudinal_sigma=longitudinal_sigma,
            interval=interval,
            spline_error=spline_error,
            **map_kwargs,
        )

    def fit_filament_right(
        self,
        image: np.ndarray,
        border: float,
        *,
        width: int = 11,
        longitudinal_sigma: float = 1.0,
        interval: float | None = None,
        spline_error: float = 0.0,
        **map_kwargs,
    ) -> Self:
        if border < 0 or 1 < border:
            raise ValueError("`border` must be in range of [0, 1].")
        spl_left = self.section((0.0, border))
        spl_right = self.section((border, 1.0))
        spl_fit = spl_right.fit_filament(
            image,
            width=width,
            degree=self.degree,
            longitudinal_sigma=longitudinal_sigma,
            interval=interval,
            spline_error=spline_error,
            **map_kwargs,
        )
        return spl_left.connect(spl_fit)

    def extend_filament_right(
        self,
        image: np.ndarray,
        length: float,
        *,
        width: int = 11,
        longitudinal_sigma: float = 1.0,
        interval: float = 1.0,
        spline_error: float = 0.0,
        **map_kwargs,
    ) -> Self:
        total_length = self.length()
        spl = self.extend_right(length, interval=interval)
        border = (total_length - 2 * interval) / (total_length + length)
        return spl.fit_filament_right(
            image,
            border,
            width=width,
            longitudinal_sigma=longitudinal_sigma,
            interval=interval,
            spline_error=spline_error,
            **map_kwargs,
        )
