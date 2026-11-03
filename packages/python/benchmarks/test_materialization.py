"""Keep the benchmark data stable when the runner's derivation changes."""

from types import SimpleNamespace

import numpy as np
from lasso import load_data, materialize


def test_explicit_dataset_seed_is_independent_of_the_run_seed(tmp_path):
    parameters = {
        "n_samples": 8,
        "n_features": 20,
        "alpha_fraction": 0.1,
        "dataset_seed": 7363749798296254,
    }
    results = []
    for seed in (123, 456):
        directory = tmp_path / str(seed)
        directory.mkdir()
        materialize(
            SimpleNamespace(seed=seed, parameters=parameters, output_dir=directory)
        )
        results.append(load_data(directory))
    for first, second in zip(results[0], results[1], strict=True):
        np.testing.assert_array_equal(first, second)
    expected = np.asfortranarray(
        np.random.default_rng(parameters["dataset_seed"]).standard_normal((8, 20))
    )
    expected -= expected.mean(axis=0)
    np.testing.assert_array_equal(results[0][0], expected)
