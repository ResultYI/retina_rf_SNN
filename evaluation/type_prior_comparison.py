from __future__ import annotations

import json
from pathlib import Path

from evaluation.type_prior_comparison_io import load_comparable_grid
from evaluation.type_prior_comparison_stats import (
    run_contract,
    stability_comparison,
    stability_summary,
    value_status,
)
from evaluation.type_prior_value_stats import (
    learning_gain_comparison,
    matched_initialization_audit,
    parameter_delta_variance,
    predictive_comparison,
)
from evaluation.type_prior_comparison_types import (
    BASE_VARIANTS,
    MIN_SEEDS,
    MIN_SOURCE_PAIRS,
    REFERENCE_VARIANT,
    SHUFFLE_VARIANTS,
    JsonMap,
    OverallStatus,
    TypePriorComparisonError,
)


def compare_type_prior_runs(paths: list[Path]) -> JsonMap:
    runs, grid = load_comparable_grid(paths)
    shuffle_variant = next(
        variant
        for variant in SHUFFLE_VARIANTS
        if any(run.variant == variant for run in runs)
    )
    required_variants = (*BASE_VARIANTS, shuffle_variant)
    identifiable = (
        len({run.seed for run in runs}) >= MIN_SEEDS
        and runs[0].source_pairs >= MIN_SOURCE_PAIRS
    )
    comparisons = {
        f"type_aware_vs_{variant}": predictive_comparison(grid, variant)
        for variant in ("type_blind", shuffle_variant)
    }
    has_reference = all(REFERENCE_VARIANT in variants for variants in grid.values())
    if has_reference:
        comparisons["type_aware_vs_cell_only"] = predictive_comparison(
            grid,
            REFERENCE_VARIANT,
        ) | {"reference_only": True}
    stability = {
        "by_variant": {
            variant: stability_summary(tuple(run for run in runs if run.variant == variant))
            for variant in required_variants
        },
        "type_aware_vs_type_blind": stability_comparison(grid, "type_blind"),
        f"type_aware_vs_{shuffle_variant}": stability_comparison(
            grid,
            shuffle_variant,
        ),
    }
    if has_reference:
        stability["by_variant"][REFERENCE_VARIANT] = stability_summary(
            tuple(run for run in runs if run.variant == REFERENCE_VARIANT)
        )
        stability["type_aware_vs_cell_only"] = stability_comparison(
            grid,
            REFERENCE_VARIANT,
        ) | {"reference_only": True}
    parameter_variance = {
        variant: parameter_delta_variance(
            tuple(run for run in runs if run.variant == variant)
        )
        for variant in required_variants
    }
    if has_reference:
        parameter_variance[REFERENCE_VARIANT] = parameter_delta_variance(
            tuple(run for run in runs if run.variant == REFERENCE_VARIANT)
        )
    predictive = value_status(
        (
            comparisons["type_aware_vs_type_blind"]["nll_delta_ci"],
            comparisons[f"type_aware_vs_{shuffle_variant}"]["nll_delta_ci"],
        ),
        identifiable=identifiable,
    )
    rf_stability = value_status(
        (
            stability["type_aware_vs_type_blind"]["cosine_delta_ci"],
            stability[f"type_aware_vs_{shuffle_variant}"]["cosine_delta_ci"],
        ),
        identifiable=identifiable,
    )
    low_budget = min({run.budget for run in runs})
    low_grid = {key: value for key, value in grid.items() if key[1] == low_budget}
    matched = runs[0].matched_initialization
    if matched:
        matched_audit = matched_initialization_audit(
            grid,
            ("type_blind", shuffle_variant),
        )
        learning = {
            "type_aware_vs_type_blind": learning_gain_comparison(
                grid,
                "type_blind",
            ),
            f"type_aware_vs_{shuffle_variant}": learning_gain_comparison(
                grid,
                shuffle_variant,
            ),
        }
        learned_stability = {
            "by_variant": {
                variant: stability_summary(
                    tuple(run for run in runs if run.variant == variant),
                    learned_delta=True,
                )
                for variant in required_variants
            },
            "type_aware_vs_type_blind": stability_comparison(
                grid,
                "type_blind",
                learned_delta=True,
            ),
            f"type_aware_vs_{shuffle_variant}": stability_comparison(
                grid,
                shuffle_variant,
                learned_delta=True,
            ),
        }
        comparisons["matched_initialization"] = matched_audit
        comparisons["learning_gain"] = learning
        comparisons["learned_delta_rf_stability"] = learned_stability
        efficiency = value_status(
            (
                learning["type_aware_vs_type_blind"]["learning_gain_delta_ci"],
                learning[f"type_aware_vs_{shuffle_variant}"]["learning_gain_delta_ci"],
            ),
            identifiable=identifiable and bool(matched_audit["passed"]),
        )
        efficiency_estimand = "learning_gain"
    else:
        efficiency = value_status(
            (
                predictive_comparison(low_grid, "type_blind")["nll_delta_ci"],
                predictive_comparison(low_grid, shuffle_variant)["nll_delta_ci"],
            ),
            identifiable=identifiable,
        )
        efficiency_estimand = "final_nll"
    return {
        "run_contract": run_contract(runs),
        "comparisons": comparisons
        | {
            "rf_stability": stability,
            "parameter_delta_variance": parameter_variance,
        },
        "predictive_value": {"status": predictive},
        "rf_stability_value": {"status": rf_stability},
        "data_efficiency_value": {
            "status": efficiency,
            "trial_budget": low_budget,
            "estimand": efficiency_estimand,
        },
        "status": overall_status((predictive, rf_stability, efficiency)),
    }


def write_type_prior_comparison(paths: list[Path], output: Path) -> None:
    report = compare_type_prior_runs(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(output)


def overall_status(statuses: tuple[str, str, str]) -> OverallStatus:
    if "not_identifiable" in statuses:
        return "not_identifiable"
    supported = "supported" in statuses
    disadvantage = "significant_disadvantage" in statuses
    if supported and disadvantage:
        return "mixed"
    if supported:
        return "supported"
    return "not_supported"


__all__ = [
    "TypePriorComparisonError",
    "compare_type_prior_runs",
    "overall_status",
    "write_type_prior_comparison",
]
