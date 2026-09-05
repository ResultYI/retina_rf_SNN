from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


AUDIT_ROOT = (
    Path(__file__).resolve().parents[1]
    / ".omo/evidence/rf-identifiability-reachability-audit"
)
if str(AUDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIT_ROOT))

from audit_artifacts import AuditContractError, strict_json_dump  # noqa: E402
from audit_decision import classify_case  # noqa: E402
from audit_design import (  # noqa: E402
    build_lagged_design,
    fit_svd_subspace,
    supported_energy_ratio,
)
from audit_oracle import fit_linear_oracle, recover_rf_weights  # noqa: E402
from audit_snn import gain_shape_decomposition  # noqa: E402
from audit_stagewise import _state_difference  # noqa: E402
from audit_teacher import (  # noqa: E402
    TeacherReference,
    analytic_conditional_rf,
    conditional_teacher_logit,
    context_basis,
)


def test_lagged_design_is_oldest_to_current_and_respects_valid_rows() -> None:
    cones = torch.arange(12, dtype=torch.float64).reshape(1, 6, 2)
    valid = torch.ones((1, 1, 6, 1), dtype=torch.bool)
    valid[:, :, 4] = False
    design, rows = build_lagged_design(cones, valid, burn_in=2, lags=3)

    assert rows == ((0, 2), (0, 3), (0, 5))
    assert torch.equal(design[0], cones[0, 0:3].reshape(-1))
    assert torch.equal(design[-1], cones[0, 3:6].reshape(-1))


def test_svd_projection_recovers_supported_and_rejects_null_energy() -> None:
    design = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    subspace = fit_svd_subspace(design, relative_threshold=1e-8, center=False)

    supported = supported_energy_ratio(torch.tensor([[2.0, -1.0, 0.0]]), subspace.basis)
    null = supported_energy_ratio(torch.tensor([[0.0, 0.0, 3.0]]), subspace.basis)
    assert supported.item() == pytest.approx(1.0)
    assert null.item() == pytest.approx(0.0)


def test_teacher_analytic_rf_matches_conditional_logit_finite_difference() -> None:
    kernel = torch.tensor(
        [[[0.6, -0.2], [0.3, 0.1], [-0.1, 0.4]]], dtype=torch.float64
    )
    sequence = torch.linspace(-1.0, 1.0, 10, dtype=torch.float64).reshape(5, 2)
    envelope = torch.tensor([1.25], dtype=torch.float64)
    history = torch.tensor([0.7], dtype=torch.float64)
    analytic = analytic_conditional_rf(kernel, envelope)
    epsilon = 1e-6
    for lag in (0, 1, 2):
        for cone in (0, 1):
            plus = sequence.clone()
            minus = sequence.clone()
            plus[-3 + lag, cone] += epsilon
            minus[-3 + lag, cone] -= epsilon
            finite = (
                conditional_teacher_logit(plus, kernel, envelope, history)
                - conditional_teacher_logit(minus, kernel, envelope, history)
            ) / (2 * epsilon)
            assert finite.item() == pytest.approx(analytic[0, lag, cone].item(), rel=1e-8)

    other_history = conditional_teacher_logit(sequence, kernel, envelope, history + 2)
    baseline = conditional_teacher_logit(sequence, kernel, envelope, history)
    assert (other_history - baseline).item() == pytest.approx(-3.0)


def test_projected_linear_oracle_recovers_rf_weights() -> None:
    generator = torch.Generator().manual_seed(7)
    design = torch.randn(80, 6, generator=generator, dtype=torch.float64)
    weights = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    targets = design @ weights + torch.tensor([0.2, -0.1, 0.3])
    subspace = fit_svd_subspace(design, relative_threshold=1e-10, center=False)
    fit = fit_linear_oracle(design, targets, subspace.basis, ridge=1e-10)
    recovered = recover_rf_weights(fit, subspace.basis)

    assert torch.allclose(recovered, weights, atol=1e-8, rtol=1e-8)
    assert fit.rmse < 1e-8


def test_context_basis_accepts_proportional_float32_envelopes() -> None:
    recovery = torch.tensor([0.0, 0.5, 0.25], dtype=torch.float64)
    amplitudes = torch.tensor([-0.2, 0.1], dtype=torch.float64)
    envelope = 1 + recovery[None, :, None] * amplitudes[None, None, :]
    envelope[0, 1, 0] += 8e-8
    reference = TeacherReference(
        torch.zeros((2, 2, 1), dtype=torch.float64),
        torch.zeros((2, 2, 1), dtype=torch.float64),
        torch.zeros((2, 2, 1), dtype=torch.float64),
        envelope,
        torch.zeros((1, 3, 1), dtype=torch.float64),
        torch.zeros((1, 1, 3, 2), dtype=torch.float64),
        torch.ones((1, 1, 3, 2), dtype=torch.bool),
        ("source",),
        ("high",),
    )

    recovered = context_basis(reference)
    assert torch.allclose(recovered, recovery / recovery.max(), atol=1e-6)


def test_gain_shape_decomposition_is_orthogonal_and_reconstructs_delta() -> None:
    before = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]])
    after = torch.tensor([[1.5, 3.0, 2.0], [1.0, 0.5, 0.5]])
    result = gain_shape_decomposition(before, after)

    assert torch.allclose(result.parallel + result.orthogonal, after - before)
    dot = (result.parallel * result.orthogonal).sum(dim=1)
    assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-6)
    assert torch.allclose(
        result.gain_energy_ratio + result.shape_energy_ratio,
        torch.ones(2),
        atol=1e-6,
    )


def test_strict_json_rejects_nonfinite_values(tmp_path: Path) -> None:
    destination = tmp_path / "payload.json"
    strict_json_dump(destination, {"finite": np.float32(1.25)})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"finite": 1.25}

    with pytest.raises(ValueError, match="finite"):
        strict_json_dump(destination, {"bad": float("nan")})


def test_state_difference_compares_sparse_tensors_exactly() -> None:
    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([2.0, 3.0])
    sparse = torch.sparse_coo_tensor(indices, values, (2, 2)).coalesce()

    maximum, mismatches = _state_difference({"sparse": sparse}, {"sparse": sparse.clone()})
    assert maximum == 0
    assert mismatches == []


def test_case_classifier_uses_predeclared_gates_and_fails_closed() -> None:
    gates = {
        "teacher_dynamic_supported_energy_at_least_0_9": True,
        "noise_free_full_cosine_at_least_0_8": True,
        "noise_free_dynamic_cosine_at_least_0_8": True,
        "sampled_full_cosine_at_least_0_8": False,
        "sampled_dynamic_cosine_at_least_0_8": False,
    }
    assert classify_case(gates) == "II"

    gates["noise_free_full_cosine_at_least_0_8"] = False
    assert classify_case(gates) == "I"

    for key in gates:
        gates[key] = True
    with pytest.raises(AuditContractError, match="Cases III-V"):
        classify_case(gates)
