"""Check the skglm adapter against an independently evaluated lasso solution."""

from types import SimpleNamespace

import numpy as np
from lasso import LassoWorker, evaluate_solution


def test_skglm_fits_the_shared_objective_without_changing_inputs():
    x = np.asfortranarray([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([2.0, -1.0])
    original_x, original_y = x.copy(), y.copy()
    context = SimpleNamespace(observation_control={"iterations": 3})
    worker = LassoWorker("skglm")
    output = worker.execute((x, y, 0.25), context)
    beta = worker.result(output, context)["coefficients"]
    np.testing.assert_allclose(beta, [1.5, -0.5])
    assert evaluate_solution(x, y, 0.25, beta).metrics["relative_gap"] <= 1e-8
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)
    statistics = worker.statistics(output, context)
    assert statistics["implementation"] == "skglm"
    assert statistics["iteration_limit"] == 3
    assert statistics["iterations"] <= 3
    assert "numba" in statistics["software"]["packages"]
