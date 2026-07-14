from __future__ import annotations

import numpy as np
import pytest
import torch

from data.cone_response import (
    ConeResponseExport,
    DataContractError,
    validate_natural_video_splits,
)
from evaluation.feasibility import (
    FeasibilityDecision,
    FeasibilityEvidence,
    assess_feasibility,
)
from evaluation.functional import (
    FunctionalAgreementTolerance,
    FunctionalResponse,
    FunctionalSummary,
    compare_functional_summaries,
    summarize_functional_response,
)
from evaluation.parameter_audit import audit_stage1_parameters
from training.stage1 import (
    MidgetSamplingMode,
    Stage1BuildConfig,
    Stage1BuildError,
    build_stage1_components,
)


def _export(source_id: str, source_kind: str) -> ConeResponseExport:
    return ConeResponseExport(
        response=np.ones((2, 1), dtype=np.float32),
        positions_degs=np.zeros((1, 2), dtype=np.float32),
        cone_types=np.zeros(1, dtype=np.uint8),
        time_axis_seconds=np.asarray([0.0, 0.005]),
        eye_trace_degs=np.zeros((2, 2), dtype=np.float32),
        units="isomerizations_per_integration_time",
        eccentricity_deg=0.0,
        source_id=source_id,
        source_movie_id=source_id,
        stimulus_source_kind=source_kind,
    )


def _positions() -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0],
         [0.4, 0.0], [0.5, 0.0], [0.6, 0.0], [0.7, 0.0]]
    )


def test_formal_video_contract_requires_natural_and_source_disjoint_splits() -> None:
    # Given
    train = (_export("movie-a", "natural_video"),)
    validation = (_export("movie-b", "natural_video"),)

    # When
    validate_natural_video_splits(train, validation)

    # Then
    with pytest.raises(DataContractError, match="source-disjoint"):
        validate_natural_video_splits(train, (_export("movie-a", "natural_video"),))
    with pytest.raises(DataContractError, match="natural_video"):
        validate_natural_video_splits(
            (_export("still-a", "parametric_image_sequence"),),
            validation,
        )


def test_nonfoveal_profile_uses_convergent_midget_sampling() -> None:
    # Given / When
    components = build_stage1_components(
        _positions(),
        Stage1BuildConfig(
            dt_ms=5.0,
            horizon_count=1,
            eccentricity_deg=2.5,
            midget_sampling=MidgetSamplingMode.CONVERGENT,
        ),
    )

    # Then
    assert components.mosaic.midget_positions_degs.shape[0] < _positions().shape[0]
    with pytest.raises(Stage1BuildError, match="private-line"):
        Stage1BuildConfig(
            dt_ms=5.0,
            horizon_count=1,
            eccentricity_deg=2.5,
            midget_sampling=MidgetSamplingMode.FOVEAL_PRIVATE_LINE,
        )


def test_parameter_audit_reports_boundary_distance_for_latent_parameters() -> None:
    # Given
    components = build_stage1_components(
        _positions(),
        Stage1BuildConfig(dt_ms=5.0, horizon_count=1),
    )

    # When
    report = audit_stage1_parameters(components)

    # Then
    names = {item.name for item in report}
    assert {
        "h1.tau_ms",
        "bipolar.tau_sustained_ms",
        "bipolar.g_ab_transient",
        "amacrine.g_ba_sustained",
        "rgc.membrane_tau_ms",
        "rgc.parasol_transient_mix",
        "decoder.fine_residual[0]",
    } <= names
    assert all(0.0 <= item.boundary_fraction <= 0.5 for item in report)


def test_functional_summary_and_go_no_go_report_use_frozen_outputs() -> None:
    # Given
    response = FunctionalResponse(
        chirp_frequency_hz=torch.tensor([0.5, 1.0, 2.0, 4.0, 8.0]),
        chirp_amplitude=torch.tensor([0.1, 0.2, 0.8, 0.4, 0.1]),
        contrast=torch.tensor([0.1, 0.2, 0.4]),
        contrast_response=torch.tensor([0.2, 0.4, 0.8]),
        grating_spatial_frequency_cpd=torch.tensor([1.0, 2.0, 4.0]),
        grating_response=torch.tensor([0.2, 0.9, 0.3]),
    )
    evidence = FeasibilityEvidence(
        structural_pass=True,
        dynamics_pass=True,
        fine_skill=0.06,
        coarse_skill=0.06,
        trained_core_skill=0.02,
        residual_gain_fraction=0.2,
        rf_agreement_fraction=0.85,
        parameters_clear_of_bounds=True,
        functional_pass=True,
    )

    # When
    summary = summarize_functional_response(response)
    report = assess_feasibility(evidence)

    # Then
    assert summary.chirp_peak_hz == 2.0
    assert summary.grating_preference_cpd == 2.0
    assert summary.contrast_gain == pytest.approx(2.0)
    assert report.decision is FeasibilityDecision.GO


def test_go_no_go_requires_finite_dual_scale_and_functional_evidence() -> None:
    # Given
    valid = dict(
        structural_pass=True,
        dynamics_pass=True,
        trained_core_skill=0.02,
        residual_gain_fraction=0.2,
        rf_agreement_fraction=0.85,
        parameters_clear_of_bounds=True,
    )

    # When
    failed_fine = assess_feasibility(
        FeasibilityEvidence(
            fine_skill=-1.0,
            coarse_skill=0.06,
            functional_pass=True,
            **valid,
        )
    )
    non_finite = assess_feasibility(
        FeasibilityEvidence(
            fine_skill=float("nan"),
            coarse_skill=0.06,
            functional_pass=True,
            **valid,
        )
    )
    missing_functional = assess_feasibility(
        FeasibilityEvidence(
            fine_skill=0.06,
            coarse_skill=0.06,
            functional_pass=False,
            **valid,
        )
    )

    # Then
    assert failed_fine.decision is FeasibilityDecision.NO_GO
    assert non_finite.decision is FeasibilityDecision.NO_GO
    assert missing_functional.decision is FeasibilityDecision.RUNS_WITHOUT_SUPPORT


def test_functional_comparison_uses_external_human_reference_tolerances() -> None:
    # Given
    model = FunctionalSummary(2.0, 2.1, 3.8)
    human_reference = FunctionalSummary(2.2, 2.0, 4.0)
    tolerance = FunctionalAgreementTolerance(
        max_frequency_octave_error=0.25,
        max_contrast_gain_relative_error=0.10,
    )

    # When
    agreement = compare_functional_summaries(
        model,
        human_reference,
        tolerance,
    )

    # Then
    assert agreement.passed
    assert agreement.contrast_gain_relative_error == pytest.approx(0.05)
