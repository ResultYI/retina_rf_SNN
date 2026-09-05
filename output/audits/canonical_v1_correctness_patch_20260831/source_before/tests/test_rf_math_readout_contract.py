from __future__ import annotations

from pathlib import Path

import pytest
import torch

from configs.physiology_profiles import macaque_photopic
from configs.rgc_type_priors import load_type_priors
from evaluation.direct_readout_paths import (
    direct_readout_intervention,
    forward_sequence_readout_paths,
)
from evaluation.response_predictions import (
    ResponsePredictionRequest,
    collect_response_predictions,
)
from models.cells.bipolar_types import BipolarConfig, BipolarConfigurationError
from models.cells.temporal import OrderedTauError, ordered_taus
from models.response_snn import build_response_retina_model
from tests.calibration_fixture import calibration_data


def test_direct_readout_paths_sum_to_canonical_and_restore_intervention() -> None:
    model = _direct_model()
    with torch.no_grad():
        model.rgc.response_bias.copy_(torch.tensor([0.1, -0.2]))
        model.rgc.bipolar_readout_gain.copy_(torch.tensor([[0.3, -0.4], [0.2, 0.1]]))
        model.rgc.amacrine_readout_gain.copy_(torch.tensor([[-0.2, 0.5], [0.1, -0.3]]))
    data = calibration_data()
    sequence = data.validation.cone_response
    history = data.validation.spike_counts[:, 0]
    canonical, _ = model.forward_sequence(sequence, observed_counts=history)
    paths, _ = forward_sequence_readout_paths(
        model,
        sequence,
        observed_counts=history,
    )
    assert torch.allclose(paths.total, canonical.spike_logits, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        paths.total,
        paths.core + paths.bipolar_direct + paths.amacrine_direct,
        atol=1e-6,
        rtol=1e-6,
    )
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with direct_readout_intervention(
        model,
        disable_bipolar=True,
        disable_amacrine=False,
    ):
        intervened, _ = forward_sequence_readout_paths(
            model,
            sequence,
            observed_counts=history,
        )
        assert torch.equal(intervened.bipolar_direct, torch.zeros_like(intervened.bipolar_direct))
    assert all(
        _tensor_equal(value, model.state_dict()[name]) for name, value in before.items()
    )


def test_thresholded_free_running_matches_legacy_alias() -> None:
    model = _direct_model()
    split = calibration_data().validation
    canonical = collect_response_predictions(
        ResponsePredictionRequest(model, split, 1, torch.device("cpu"), "thresholded_free_running")
    )
    legacy = collect_response_predictions(
        ResponsePredictionRequest(model, split, 1, torch.device("cpu"), "free_running")
    )
    assert torch.equal(canonical.logits, legacy.logits)
    assert torch.equal(canonical.generator_potential, legacy.generator_potential)


def test_ordered_taus_preserve_strict_runtime_order() -> None:
    bounds = torch.tensor(((20.0, 200.0), (5.0, 120.0)))
    taus = ordered_taus(torch.tensor(0.2), torch.tensor(-0.4), bounds)
    assert 5.0 < float(taus[1]) < float(taus[0]) < 200.0


def test_ordered_taus_reject_unsafe_bounds_without_remapping() -> None:
    bounds = torch.tensor(((20.0, 20.0), (5.0, 30.0)))
    with pytest.raises(OrderedTauError, match="sustained tau lower"):
        ordered_taus(torch.tensor(0.0), torch.tensor(0.0), bounds)


def test_bipolar_config_rejects_bounds_without_ordered_interval() -> None:
    values = _bipolar_config_values()
    values["tau_sustained_min_ms"] = 5.0
    values["tau_sustained_max_ms"] = 10.0
    values["initial_tau_sustained_ms"] = 9.0
    values["tau_transient_min_ms"] = 10.0
    values["tau_transient_max_ms"] = 20.0
    values["initial_tau_transient_ms"] = 11.0
    with pytest.raises(BipolarConfigurationError):
        BipolarConfig(**values)


def test_math_document_uses_total_dynamic_16_lag_contract() -> None:
    text = Path("docs/mathematical_formulation/retina_rf_snn_formulas_zh.tex").read_text(
        encoding="utf-8"
    )
    assert "conditional total-dynamic logit RF" in text
    assert "L_{\\mathrm{RF}}=16" in text
    assert "thresholded\\_free\\_running" in text
    assert "fixed physiological state" not in text


def _direct_model():
    data = calibration_data()
    priors = load_type_priors(
        "configs/rgc_type_priors.yaml",
        required_type_ids=("midget", "parasol"),
    )
    return build_response_retina_model(
        torch.as_tensor(data.cone_positions_degs),
        data.cells,
        macaque_photopic(dt_ms=5.0, cone_spacing_deg=0.1, eccentricity_deg=4.0),
        priors,
        support_radius_degs=0.2,
        readout_rate_tau_ms=50.0,
        surrogate_slope=5.0,
        parameter_sharing_mode="type_blind",
        enable_response_bias=True,
        enable_direct_readout=True,
    )


def _bipolar_config_values() -> dict[str, float]:
    return {
        "dt_ms": 5.0,
        "initial_tau_sustained_ms": 80.0,
        "tau_sustained_min_ms": 20.0,
        "tau_sustained_max_ms": 200.0,
        "initial_tau_transient_ms": 20.0,
        "tau_transient_min_ms": 5.0,
        "tau_transient_max_ms": 120.0,
        "initial_g_ab_sustained": 0.01,
        "g_ab_sustained_max": 0.1,
        "initial_g_ab_transient": 0.01,
        "g_ab_transient_max": 0.3,
    }


def _tensor_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.is_sparse or right.is_sparse:
        a = left.coalesce()
        b = right.coalesce()
        return torch.equal(a.indices(), b.indices()) and torch.equal(a.values(), b.values())
    return torch.equal(left, right)
