import numpy as np
import pytest
from numpy.testing import assert_allclose

from napari_filaments import _optimizer as _opt


@pytest.mark.parametrize(
    ["optimizer", "params"],
    [
        (_opt.GaussianOptimizer, [8.0, 1.0, 1.0, 0.2]),
        (_opt.ErfOptimizer, [8.0, 1.0, 1.0, 0.2]),
        (_opt.TwosideErfOptimizer, [5.0, 18.0, 1.0, 1.0, 0.2, 0.3]),
    ],
)
def test_optimization(optimizer: "type[_opt.Optimizer]", params):
    opt = optimizer(params)
    x = np.arange(21)
    y = opt.sample(x) + np.random.normal(scale=0.1, size=21)
    opt_optimized = opt.optimize(y)
    opt_fit = optimizer.fit(y)
    assert_allclose(opt_optimized.params, opt_fit.params, rtol=1e-4, atol=1e-4)
