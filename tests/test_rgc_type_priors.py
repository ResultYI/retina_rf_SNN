from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from configs.rgc_type_priors import TypePriorConfigurationError, load_type_priors


def _prior() -> dict[str, dict[str, float]]:
    parameter = {"mean": 0.5, "lower": 0.1, "upper": 1.0}
    return {
        "spatial_sigma": parameter,
        "sustained_mix": {"mean": 0.5, "lower": 0.0, "upper": 1.0},
        "membrane_tau_ms": {"mean": 20.0, "lower": 5.0, "upper": 250.0},
        "adaptation_tau_ms": {"mean": 80.0, "lower": 5.0, "upper": 250.0},
        "adaptation_gain": parameter,
        "amacrine_gain": parameter,
        "threshold": parameter,
        "subunit_tau_ms": {"mean": 50.0, "lower": 5.0, "upper": 250.0},
        "subunit_gain": parameter,
    }


def test_loads_overlapping_soft_type_priors(tmp_path: Path) -> None:
    path = tmp_path / "priors.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "cell_residual_scale": 0.25,
                "cell_residual_weight": 0.01,
                "type_prior_weight": 0.02,
                "types": {"midget": _prior(), "parasol": _prior()},
            }
        ),
        encoding="utf-8",
    )

    priors = load_type_priors(path, required_type_ids=("midget", "parasol"))

    assert priors.cell_residual_scale == 0.25
    assert priors.type_prior_weight == 0.02
    assert set(priors.type_ids) == {"midget", "parasol"}


def test_rejects_missing_type_coverage(tmp_path: Path) -> None:
    path = tmp_path / "priors.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "cell_residual_scale": 0.25,
                "cell_residual_weight": 0.01,
                "type_prior_weight": 0.02,
                "types": {"midget": _prior()},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypePriorConfigurationError, match="coverage"):
        load_type_priors(path, required_type_ids=("midget", "parasol"))
