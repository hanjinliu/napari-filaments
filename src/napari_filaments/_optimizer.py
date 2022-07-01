from __future__ import annotations

from abc import ABC, abstractstaticmethod
from typing import TYPE_CHECKING, NamedTuple, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit
from scipy.special import erf

if TYPE_CHECKING:
    from typing_extensions import Self


sq2 = np.sqrt(2)
Bounds = Tuple[Union[ArrayLike, float], Union[ArrayLike, float]]


class Optimizer(ABC):
    @abstractstaticmethod
    class Parameters(NamedTuple):
        pass

    def __init__(
        self, params=None, cov=None, bounds: Bounds = (-np.inf, np.inf)
    ):
        self.params = params
        self.cov = cov
        self.bounds = bounds

    @property
    def params(self) -> Parameters:
        return self._params

    @params.setter
    def params(self, val):
        if val is None:
            self._params = None
        else:
            self._params = self.Parameters(*val)

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
        for ydata in arr:
            params, bounds = cls.initialize(ydata)
            params, _ = curve_fit(cls.model, x, ydata, params, bounds=bounds)
            results.append(params)
        return np.stack(results, axis=0)

    def sample(self, xdata: np.ndarray) -> np.ndarray:
        """Sample points at given x coordinates."""
        return self.model(xdata, *self.params)

    @abstractstaticmethod
    def model(xdata: np.ndarray, *args: Parameters):
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
    class Parameters(NamedTuple):
        mu: float
        sg: float
        a: float
        b: float

    if TYPE_CHECKING:

        @property
        def params(self) -> Parameters:
            ...

    @staticmethod
    def model(xdata: np.ndarray, mu, sg, a, b):
        return a * np.exp(-((xdata - mu) ** 2) / (2 * sg**2)) + b

    @staticmethod
    def initialize(ydata: np.ndarray) -> tuple[ArrayLike, Bounds]:
        ymin = np.min(ydata)
        argmax = np.argmax(ydata)
        ymax = ydata[argmax]
        dy = ymax - ymin
        bounds = (
            [0.0, 0.0, 0.0, ymin],
            [ydata.size, np.inf, dy * 1.1, ymax],
        )
        params = [argmax, 2.0, dy, ymin + dy * 0.05]
        return params, bounds


class ErfOptimizer(Optimizer):
    class Parameters(NamedTuple):
        mu: float
        sg: float
        a: float
        b: float

    if TYPE_CHECKING:

        @property
        def params(self) -> Parameters:
            ...

    @staticmethod
    def model(xdata: np.ndarray, mu, sg, a, b):
        x0 = (xdata - mu) / sg / sq2
        return (a - b) / 2 * (1 + erf(x0)) + b

    @staticmethod
    def initialize(ydata: np.ndarray) -> tuple[ArrayLike, Bounds]:
        ymin = np.min(ydata)
        ymax = np.max(ydata)
        ndata = ydata.size
        a = np.mean(ydata[-3:])
        b = np.mean(ydata[:3])
        params = [ndata, 2.0, a, b]
        bounds = (
            [0.0, 0.0, 0.0, ymin],
            [ndata, np.inf, ymax - ymin, ymax],
        )
        return params, bounds


class TwosideErfOptimizer(Optimizer):
    r"""
          a ______________
           /              \_____ b1
    b0 ___/
          mu1            mu2
    """

    class Parameters(NamedTuple):
        mu0: float
        mu1: float
        sg0: float
        sg1: float
        a: float
        b0: float
        b1: float

    if TYPE_CHECKING:

        @property
        def params(self) -> Parameters:
            ...

    @staticmethod
    def model(xdata: np.ndarray, mu0, mu1, sg0, sg1, a, b0, b1):
        return ErfOptimizer.model(xdata, mu0, sg0, a, b0) - ErfOptimizer.model(
            xdata, mu1, sg1, a - b1, 0
        )

    @staticmethod
    def initialize(ydata: np.ndarray) -> tuple[ArrayLike, Bounds]:
        ndata = ydata.size
        xc = ndata // 2
        ymin_l = np.min(ydata[:xc])
        ymin_r = np.min(ydata[xc:])
        ymin = min(ymin_l, ymin_r)
        ymax = np.max(ydata)
        params = [2.0, ndata - 2, 2.0, 2.0, ymax, ymin_l, ymin_r]
        bounds = (
            [0.0, 0.0, 0.0, 0.0, 0.0, ymin, ymin],
            [ndata, ndata, np.inf, np.inf, ymax - ymin, ymax, ymax],
        )
        return params, bounds
