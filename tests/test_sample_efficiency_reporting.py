from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import math
from pathlib import Path

from PIL import Image
import pytest

from evaluation.model_comparison.sample_efficiency_presentation import (
    build_decision_payload,
    sample_efficiency_report,
    write_sample_efficiency_figures,
)
from evaluation.model_comparison.sample_efficiency_reporting import (
    MetricRow,
    Profile,
    ReportingSchemaError,
    aggregate_metric_rows,
)


@dataclass(frozen=True, slots=True)
class RowCase:
    model: str
    profile: Profile
    ce: float
    rf: float | None
    exact_cell: float | None
    fraction: float = 0.25
    bank_seed: int = 31001
    model_seed: int | None = 19
    params: int = 112
    source: str = "new"


def _row(case: RowCase) -> MetricRow:
    return MetricRow(
        fraction=case.fraction,
        train_stimuli=int(112 * case.fraction),
        model=case.model,
        profile=case.profile,
        bank_seed=case.bank_seed,
        model_seed=case.model_seed,
        parameter_count=case.params,
        val_ce=case.ce,
        sampled_nll=case.ce + 0.1,
        bits_per_spike=0.7 - case.ce,
        global_rf=case.rf,
        spatial_rf=None if case.rf is None else case.rf - 0.01,
        temporal_rf=None if case.rf is None else case.rf - 0.02,
        exact_cell=case.exact_cell,
        nearest_type_polarity=None if case.rf is None else 1.0,
        prototype_centroid=None if case.rf is None else 0.95,
        cross_seed_rf=None if case.rf is None else case.rf - 0.03,
        cross_bank_rf=None if case.rf is None else case.rf - 0.04,
        reuse_status="trained",
        source_run_id=case.source,
    )


def test_aggregate_metric_rows_when_profiles_and_fractions_are_valid() -> None:
    rows = (
        _row(
            RowCase(
                "Mechanistic Retina",
                Profile.ARCHITECTURE_SIZE,
                0.30,
                0.80,
                0.25,
                params=264,
            )
        ),
        _row(
            RowCase(
                "Mechanistic Retina",
                Profile.ARCHITECTURE_SIZE,
                0.34,
                0.90,
                0.50,
                bank_seed=31002,
                params=264,
            )
        ),
        _row(
            RowCase(
                "Bias",
                Profile.SHARED_REFERENCE,
                0.45,
                None,
                None,
                model_seed=None,
                params=16,
            )
        ),
        _row(RowCase("LN-LN", Profile.ACTIVE_DOF, 0.40, 0.60, 0.10, params=160)),
    )

    aggregates = aggregate_metric_rows(rows)

    mechanistic = next(
        row
        for row in aggregates
        if row.model == "Mechanistic Retina"
        and row.profile is Profile.ARCHITECTURE_SIZE
    )
    bias = next(row for row in aggregates if row.model == "Bias")
    assert mechanistic.runs == 2
    assert mechanistic.val_ce_mean == pytest.approx(0.32)
    assert mechanistic.val_ce_sd == pytest.approx(0.0282842712)
    assert mechanistic.global_rf_mean == pytest.approx(0.85)
    assert mechanistic.cross_seed_rf_mean == pytest.approx(0.82)
    assert mechanistic.cross_bank_rf_mean == pytest.approx(0.81)
    assert mechanistic.exact_cell_mean == pytest.approx(0.375)
    assert mechanistic.nearest_type_polarity_mean == pytest.approx(1.0)
    assert bias.global_rf_mean is None
    assert bias.exact_cell_mean is None


def test_aggregate_metric_rows_when_glm_has_no_model_seed_allows_no_cross_seed() -> (
    None
):
    row = replace(
        _row(
            RowCase(
                "GLM-SH",
                Profile.SHARED_REFERENCE,
                0.37,
                0.12,
                0.0,
                model_seed=None,
                params=7504,
            )
        ),
        cross_seed_rf=None,
        cross_bank_rf=0.11,
    )

    aggregate = aggregate_metric_rows((row,))[0]

    assert aggregate.model == "GLM-SH"
    assert aggregate.global_rf_mean == pytest.approx(0.12)
    assert aggregate.cross_seed_rf_mean is None
    assert aggregate.cross_bank_rf_mean == pytest.approx(0.11)


def test_write_sample_efficiency_figures_when_rows_are_valid(tmp_path: Path) -> None:
    rows = (
        _row(
            RowCase(
                "Mechanistic Retina",
                Profile.ARCHITECTURE_SIZE,
                0.30,
                0.80,
                0.25,
                params=264,
            )
        ),
        _row(
            RowCase(
                "Mechanistic Retina",
                Profile.ARCHITECTURE_SIZE,
                0.28,
                0.84,
                0.50,
                fraction=0.50,
                params=264,
            )
        ),
        _row(
            RowCase(
                "Bias",
                Profile.SHARED_REFERENCE,
                0.45,
                None,
                None,
                model_seed=None,
                params=16,
            )
        ),
    )

    write_sample_efficiency_figures(tmp_path, rows)

    assert {path.name for path in tmp_path.glob("*.png")} == {
        "sample-efficiency-ce.png",
        "sample-efficiency-rf.png",
    }
    for name in ("sample-efficiency-rf.png", "sample-efficiency-ce.png"):
        with Image.open(tmp_path / name) as image:
            image.verify()
        assert (tmp_path / name).stat().st_size > 0


def test_decision_payload_when_mechanistic_has_mixed_regime_results() -> None:
    rows = (
        _row(
            RowCase(
                "Mechanistic Retina",
                Profile.ARCHITECTURE_SIZE,
                0.40,
                0.90,
                0.60,
                fraction=1.0,
                params=264,
            )
        ),
        _row(
            RowCase(
                "Graph-TCN",
                Profile.ARCHITECTURE_SIZE,
                0.30,
                0.70,
                0.20,
                fraction=1.0,
                params=224,
            )
        ),
        _row(
            RowCase(
                "Mechanistic Retina",
                Profile.ACTIVE_DOF,
                0.25,
                0.80,
                0.50,
                fraction=1.0,
                params=264,
            )
        ),
        _row(
            RowCase(
                "LN-LN", Profile.ACTIVE_DOF, 0.35, 0.82, 0.10, fraction=1.0, params=160
            )
        ),
    )

    payload = build_decision_payload(aggregate_metric_rows(rows))
    report = sample_efficiency_report(aggregate_metric_rows(rows), payload)

    regimes = payload["regimes"]
    assert isinstance(regimes, list)
    assert [entry["profile"] for entry in regimes] == [
        "architecture-size",
        "optimizer-listed-count",
    ]
    assert [entry["outcome"] for entry in regimes] == ["mixed", "mixed"]
    assert "negative and mixed outcomes are allowed" in report
    assert "architecture-size" in report
    assert "optimizer-listed-count" in report


def test_aggregate_metric_rows_when_duplicate_identity_fails() -> None:
    first = _row(RowCase("Mechanistic Retina", Profile.ACTIVE_DOF, 0.30, 0.80, 0.25))

    with pytest.raises(ReportingSchemaError, match="DUPLICATE_METRIC_IDENTITY"):
        aggregate_metric_rows((first, first))


def test_aggregate_metric_rows_when_missing_stability_fields_fails() -> None:
    row = _row(RowCase("Mechanistic Retina", Profile.ACTIVE_DOF, 0.30, 0.80, 0.25))
    missing_cross_bank = replace(row, cross_bank_rf=None)

    with pytest.raises(ReportingSchemaError, match="MISSING_VARIATION_FIELD"):
        aggregate_metric_rows((missing_cross_bank,))


def test_aggregate_metric_rows_when_mixed_regime_row_fails() -> None:
    row = _row(
        RowCase(
            "Bias", Profile.ACTIVE_DOF, 0.45, None, None, model_seed=None, params=16
        )
    )

    with pytest.raises(ReportingSchemaError, match="MIXED_REGIME_ROW"):
        aggregate_metric_rows((row,))


@pytest.mark.parametrize(
    ("row", "field_name"),
    (
        (
            replace(
                _row(
                    RowCase("Mechanistic Retina", Profile.ACTIVE_DOF, 0.30, 0.80, 0.25)
                ),
                val_ce=math.nan,
            ),
            "val_ce",
        ),
        (
            replace(
                _row(
                    RowCase("Mechanistic Retina", Profile.ACTIVE_DOF, 0.30, 0.80, 0.25)
                ),
                sampled_nll=math.inf,
            ),
            "sampled_nll",
        ),
        (
            replace(
                _row(
                    RowCase("Mechanistic Retina", Profile.ACTIVE_DOF, 0.30, 0.80, 0.25)
                ),
                bits_per_spike=-math.inf,
            ),
            "bits_per_spike",
        ),
    ),
)
def test_aggregate_metric_rows_when_prediction_metric_is_nonfinite_fails(
    row: MetricRow, field_name: str
) -> None:
    with pytest.raises(
        ReportingSchemaError, match=f"NONFINITE_METRIC_VALUE: {field_name}"
    ):
        aggregate_metric_rows((row,))


@pytest.mark.parametrize(
    ("row", "field_name"),
    (
        (
            replace(
                _row(
                    RowCase("Mechanistic Retina", Profile.ACTIVE_DOF, 0.30, 0.80, 0.25)
                ),
                global_rf=math.nan,
            ),
            "global_rf",
        ),
        (
            replace(
                _row(
                    RowCase("Mechanistic Retina", Profile.ACTIVE_DOF, 0.30, 0.80, 0.25)
                ),
                spatial_rf=math.inf,
            ),
            "spatial_rf",
        ),
        (
            replace(
                _row(
                    RowCase("Mechanistic Retina", Profile.ACTIVE_DOF, 0.30, 0.80, 0.25)
                ),
                temporal_rf=-math.inf,
            ),
            "temporal_rf",
        ),
        (
            replace(
                _row(
                    RowCase("Mechanistic Retina", Profile.ACTIVE_DOF, 0.30, 0.80, 0.25)
                ),
                exact_cell=math.nan,
            ),
            "exact_cell",
        ),
        (
            replace(
                _row(
                    RowCase("Mechanistic Retina", Profile.ACTIVE_DOF, 0.30, 0.80, 0.25)
                ),
                nearest_type_polarity=math.inf,
            ),
            "nearest_type_polarity",
        ),
        (
            replace(
                _row(
                    RowCase("Mechanistic Retina", Profile.ACTIVE_DOF, 0.30, 0.80, 0.25)
                ),
                prototype_centroid=-math.inf,
            ),
            "prototype_centroid",
        ),
        (
            replace(
                _row(
                    RowCase("Mechanistic Retina", Profile.ACTIVE_DOF, 0.30, 0.80, 0.25)
                ),
                cross_seed_rf=math.nan,
            ),
            "cross_seed_rf",
        ),
        (
            replace(
                _row(
                    RowCase("Mechanistic Retina", Profile.ACTIVE_DOF, 0.30, 0.80, 0.25)
                ),
                cross_bank_rf=math.inf,
            ),
            "cross_bank_rf",
        ),
    ),
)
def test_aggregate_metric_rows_when_present_rf_or_stability_metric_is_nonfinite_fails(
    row: MetricRow, field_name: str
) -> None:
    with pytest.raises(
        ReportingSchemaError, match=f"NONFINITE_METRIC_VALUE: {field_name}"
    ):
        aggregate_metric_rows((row,))
