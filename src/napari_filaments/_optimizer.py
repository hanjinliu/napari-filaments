from __future__ import annotations

from abc import ABC, abstractstaticmethod
from typing import TYPE_CHECKING, Union

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf as sp_erf

if TYPE_CHECKING:
    from typing_extensions import Self


sq2 = np.sqrt(2)
Bounds = tuple[Union[np.ndarray, float], Union[np.ndarray, float]]


class Optimizer(ABC):
    def __init__(
        self, params=None, cov=None, bounds: Bounds = (-np.inf, np.inf)
    ):
        self.params = params
        self.cov = cov
        self.bounds = bounds

    def optimize(self, ydata: np.ndarray) -> Self:
        """Fit data to the model function and return a new instance."""
        xdata = np.arange(ydata.size)
        if self.params is None:
            self.params, self.bounds = self.initialize(ydata)
        params, cov = curve_fit(
            self.model, xdata, ydata, self.params, bounds=self.bounds
        )
        return self.__class__(params, cov, self.bounds)

    @classmethod
    def multi_optimize(cls, arr: np.ndarray) -> np.ndarray:
        x = np.arange(arr.shape[1])
        results: list[np.ndarray] = []
        for i, ydata in enumerate(arr):
            params, bounds = cls.initialize(ydata)
            params, cov = curve_fit(cls.model, x, ydata, params, bounds=bounds)
            results.append(params)
        return np.stack(results, axis=0)

    def sample(self, xdata: np.ndarray) -> np.ndarray:
        """Sample points at given x coordinates."""
        return self.model(xdata, *self.params)

    @abstractstaticmethod
    def model(xdata: np.ndarray, *args):
        """Model function."""

    @abstractstaticmethod
    def initialize(ydata: np.ndarray) -> tuple[np.ndarray, Bounds]:
        """Initialize parameters and bounds."""

    @classmethod
    def fit(cls, ydata: np.ndarray) -> Self:
        """Construct from data."""
        params, bounds = cls.initialize(ydata)
        return cls(params, bounds).optimize(ydata)


class GaussianOptimizer(Optimizer):
    @staticmethod
    def model(xdata: np.ndarray, mu, sg, a, b):
        return a * np.exp(-((xdata - mu) ** 2) / (2 * sg**2)) + b

    @staticmethod
    def initialize(ydata: np.ndarray) -> tuple[np.ndarray, Bounds]:
        bounds = (
            [0.0, 0.0, 0.0, -np.inf],
            [ydata.size, np.inf, np.inf, np.inf],
        )
        argmax = np.argmax(ydata)
        params = np.array([argmax, 2.0, ydata[argmax], 0.0])
        return params, bounds


class ErfOptimizer(Optimizer):
    @staticmethod
    def model(xdata: np.ndarray, mu, sg, a, b):
        x0 = (xdata - mu) / sg
        return (a - b) / 2 * (1 + sp_erf(x0) / sq2) + b

    @staticmethod
    def initialize(ydata: np.ndarray) -> tuple[np.ndarray, Bounds]:
        ndata = ydata.size
        a = np.mean(ydata[-3:])
        b = np.mean(ydata[:3])
        params = np.array([ndata, 2.0, a, b])
        bounds = (
            [0, 0, -np.inf, -np.inf],
            [ndata, np.inf, np.inf, np.inf],
        )
        return params, bounds


class TwosideErfOptimizer(Optimizer):
    r"""
          a ______________
           /              \_____ b1
    b0 ___/
          mu1            mu2
    """

    @staticmethod
    def model(xdata: np.ndarray, mu0, mu1, sg, a, b0, b1):
        return ErfOptimizer.model(xdata, mu0, sg, a, b0) - ErfOptimizer.model(
            xdata, mu1, sg, a - b1, 0
        )

    @staticmethod
    def initialize(ydata: np.ndarray) -> tuple[np.ndarray, Bounds]:
        ndata = ydata.size
        xc = ndata // 2
        params = np.array(
            [
                2,
                ndata - 2,
                2.0,
                np.max(ydata),
                np.min(ydata[:xc]),
                np.min(ydata[xc:]),
            ]
        )
        bounds = (
            [0.0, 0, 0.0, -np.inf, -np.inf, -np.inf],
            [ndata, ndata, np.inf, np.inf, np.inf, np.inf],
        )
        return params, bounds
