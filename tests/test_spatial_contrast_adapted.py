from __future__ import annotations

import importlib.util
import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import approx_fprime

from baselines.center_surround_ln import CenterSurroundLN


def test_given_frozen_definition_when_resolved_then_adapter_exists() -> None:
    assert importlib.util.find_spec("baselines.spatial_contrast_adapted") is not None


def test_given_ln_center_when_recovered_then_signed_scaled_activation_matches() -> None:
    from baselines.spatial_contrast_adapted import CenterFilter, features_for_sequences

    model = CenterSurroundLN(1000 / 150, 30, 61001)
    with torch.no_grad():
        model.raw_amplitudes[0] = 2.3
        model.raw_temporal[0] *= -1
    cones = torch.randn(2, 80, 289, generator=torch.Generator().manual_seed(12))
    frozen = CenterFilter.from_ln(model)
    features, verification = features_for_sequences(frozen, cones)
    assert features.shape == (2, 80, 2)
    assert verification["center_activation_double_max_abs_error"] < 1e-10
    assert verification["center_activation_native_max_abs_error"] < 1e-5
    np.testing.assert_array_equal(frozen.spatial, model.spatial_components()[0].detach().numpy())
    np.testing.assert_array_equal(frozen.temporal, model.temporal_kernels()[0].detach().numpy())


def test_given_constant_space_when_filtered_then_contrast_is_zero_and_causal() -> None:
    from baselines.spatial_contrast_adapted import CenterFilter, features_for_sequences

    model = CenterSurroundLN(1000 / 150, 30, 61001)
    cones = torch.zeros(1, 80, 289)
    cones[:, 31:] = 2
    features, _ = features_for_sequences(CenterFilter.from_ln(model), cones)
    np.testing.assert_array_equal(features[:, :31], 0)
    np.testing.assert_allclose(features[:, :, 1], 0, atol=1e-14)


def test_given_fit_mask_when_standardizing_then_only_fitting_bins_are_used() -> None:
    from baselines.spatial_contrast_adapted import FittingData

    features = np.array([[[0., 1.], [2., 3.], [1e9, -1e9]]])
    counts = np.array([[[0], [2], [999]]])
    mask = np.array([[[True], [True], [False]]])
    fitting = FittingData.from_arrays(features, counts, mask)
    np.testing.assert_array_equal(fitting.mean, [1., 2.])
    np.testing.assert_array_equal(fitting.std, [1., 1.])
    np.testing.assert_array_equal(fitting.events, [0., 1.])
    assert fitting.maximum_count == 999


def test_given_expected_counts_when_scored_then_bernoulli_link_and_gradient_match() -> None:
    from baselines.spatial_contrast_adapted import FittingData, bernoulli_objective
    from baselines.spatial_contrast_official import vectorized_softplus

    features = np.array([[[0., 1.], [2., 4.], [1., 0.], [3., 2.]]])
    counts = np.array([[[0], [2], [1], [0]]])
    fitting = FittingData.from_arrays(features, counts, np.ones_like(counts, dtype=bool))
    parameters = np.array([0.8, -0.3, 0.5, 0.7])
    loss, gradient = bernoulli_objective(parameters, fitting)
    rate = vectorized_softplus(fitting.standardized, parameters)
    p = -np.expm1(-rate)
    expected = -(fitting.events * np.log(p) + (1-fitting.events) * np.log1p(-p)).sum()
    assert loss == pytest.approx(expected / fitting.events.sum())
    numerical = approx_fprime(parameters, lambda value: bernoulli_objective(value, fitting)[0], 1e-6)
    np.testing.assert_allclose(gradient, numerical, atol=3e-6, rtol=3e-5)


def test_given_w_zero_when_contrast_changes_then_prediction_is_identical() -> None:
    from baselines.spatial_contrast_official import vectorized_softplus

    features = np.array([[1., 2., 3.], [0.5, 0.2, 1.]])
    altered = features.copy()
    altered[1] += 100
    params = np.array([1., -2., 1., 0.])
    np.testing.assert_array_equal(vectorized_softplus(features, params), vectorized_softplus(altered, params))
    np.testing.assert_array_equal(vectorized_softplus(features, params), vectorized_softplus(features[:1], params[:3]))


def test_given_official_snapshot_when_vendored_then_function_bodies_are_unchanged() -> None:
    root = Path(__file__).resolve().parents[1]
    vendor = ast.parse((root / "baselines/spatial_contrast_official.py").read_text())
    for source, names in (
        ("convolutions", ("convolve_stimulus_with_kernels_for_sc", "_get")),
        ("nonlinearities", ("vectorized_softplus", "vectorized_softplus_derivative")),
    ):
        path = root / f".omo/evidence/spatial_contrast_baseline/sources/sc_model__utils__{source}.py.txt"
        original = ast.parse(path.read_text())
        for name in names:
            expected = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == name)
            actual = next(node for node in vendor.body if isinstance(node, ast.FunctionDef) and node.name == name)
            assert [ast.dump(node) for node in actual.body] == [ast.dump(node) for node in expected.body]


def test_given_unrecoverable_center_when_loaded_then_execution_stops() -> None:
    from baselines.center_surround_ln import LNError
    from baselines.spatial_contrast_adapted import CenterFilter

    model = CenterSurroundLN(1000 / 150, 30, 61001)
    with torch.no_grad():
        model.raw_temporal[0].fill_(0)
    with pytest.raises(LNError):
        CenterFilter.from_ln(model)
