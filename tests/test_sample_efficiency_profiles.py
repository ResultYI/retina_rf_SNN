from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import pytest

from baselines.graph_tcn import graph_tcn_parameter_count, select_hidden_width
from baselines.lnln_subunit import lnln_parameter_count, select_subunit_count
import evaluation.model_comparison.sample_efficiency_profiles as profiles_module
from evaluation.model_comparison.sample_efficiency_profiles import (
    FairnessRegime,
    InvalidCapacityClaimError,
    MechanisticInventory,
    ProfileSourceMismatchError,
    claim_exact_active_dof_match,
    format_parameter_report,
    profile_sets,
)


def test_profiles_encode_predeclared_counts_and_honest_labels() -> None:
    # Given: the predeclared Candidate0 T=2 fairness profiles.
    sets = profile_sets()

    # When: profiles are read by fairness regime.
    architecture = sets.architecture_size.profile_named("LN-LN")
    graph_size = sets.architecture_size.profile_named("Graph-TCN")
    active_lnln = sets.active_dof.profile_named("LN-LN")
    active_graph = sets.active_dof.profile_named("Graph-TCN")
    shared = (
        sets.architecture_size.profile_named("Bias"),
        sets.architecture_size.profile_named("GLM-SH"),
        sets.active_dof.profile_named("Bias"),
        sets.active_dof.profile_named("GLM-SH"),
    )

    # Then: all counts and labels are deterministic and honest.
    assert sets.mechanistic == MechanisticInventory(
        total=264,
        requires_grad=264,
        optimizer_listed=136,
        nonzero_gradient=None,
        actually_updated=None,
    )
    assert architecture.selected_value == 2
    assert architecture.parameter_count == 240
    assert graph_size.selected_value == 5
    assert graph_size.parameter_count == 270
    assert active_lnln.selected_value == 1
    assert active_lnln.parameter_count == 160
    assert active_lnln.label == (
        "nearest-feasible over-target optimizer-listed-count control"
    )
    assert active_graph.selected_value == 2
    assert active_graph.parameter_count == 144
    assert active_graph.label == "near optimizer-listed-count matched"
    assert all(
        profile.label == "shared/unmatched reference baseline" for profile in shared
    )


def test_profile_selection_is_deterministic_and_uses_no_result_inputs() -> None:
    # Given: profile construction is invoked repeatedly.
    first = profile_sets()

    # When: the same contract is built again.
    second = profile_sets()

    # Then: the result is byte-stable and the selection API has no result input.
    assert first == second
    assert first.architecture_size.regime is FairnessRegime.ARCHITECTURE_SIZE
    assert second.active_dof.regime is FairnessRegime.ACTIVE_DOF
    signature = inspect.signature(profile_sets)
    assert tuple(signature.parameters) == ()
    source = inspect.getsource(profile_sets)
    assert "sweep" not in source.lower()
    assert "validation" not in source.lower()
    assert "result" not in source.lower()


def test_profile_builder_source_excludes_forbidden_dependencies() -> None:
    # Given: the production module and private builder are inspected directly.
    source = inspect.getsource(profiles_module)
    builder = inspect.getsource(profiles_module._build_profile_sets)
    imports = tuple(
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    )

    # When: the full implementation path is checked for forbidden dependencies.
    forbidden_tokens = ("result", "validation", "sweep", "random", "training")
    builder_calls = (
        "select_subunit_count(MECHANISTIC_TOTAL, CELL_COUNT, GROUP_COUNT)",
        "MECHANISTIC_OPTIMIZER_LISTED",
        "select_hidden_width(MECHANISTIC_TOTAL, CELL_COUNT)",
        "select_hidden_width(MECHANISTIC_OPTIMIZER_LISTED, CELL_COUNT)",
    )

    # Then: selection depends only on deterministic selectors and count functions.
    assert imports == (
        "from __future__ import annotations",
        "from dataclasses import dataclass",
        "from enum import StrEnum",
        "from typing import Final",
        "from baselines.graph_tcn import graph_tcn_parameter_count, select_hidden_width",
        "from baselines.lnln_subunit import lnln_parameter_count, select_subunit_count",
    )
    assert all(token not in source.lower() for token in forbidden_tokens)
    assert all(token not in builder.lower() for token in forbidden_tokens)
    assert all(call in builder for call in builder_calls)
    assert "lnln_parameter_count(" in builder
    assert "graph_tcn_parameter_count(" in builder


def test_selectors_keep_confirmed_tie_breaking_targets() -> None:
    # Given: manually confirmed equal-distance targets for active controls.
    lnln_target = 200
    graph_target = 127
    lnln_one = lnln_parameter_count(1, 16, 4)
    lnln_two = lnln_parameter_count(2, 16, 4)
    graph_one = graph_tcn_parameter_count(1, 16)
    graph_two = graph_tcn_parameter_count(2, 16)

    # When: the existing deterministic selectors resolve exact ties.
    lnln_selected = select_subunit_count(lnln_target, 16, 4)
    graph_selected = select_hidden_width(graph_target, 16)

    # Then: ties break toward the lower feasible model size.
    assert abs(lnln_one - lnln_target) == abs(lnln_two - lnln_target)
    assert abs(graph_one - graph_target) == abs(graph_two - graph_target)
    assert lnln_selected == 1
    assert graph_selected == 1


def test_parameter_report_uses_five_honest_parameter_categories() -> None:
    # Given: the optimizer-listed-count profile contains a 160-parameter LN-LN control.
    sets = profile_sets()

    # When: a human-readable parameter report is produced.
    report = format_parameter_report(sets)

    # Then: it does not call 264 effective trainable DoF or LN-LN 160 exact.
    assert (
        "Mechanistic inventory: total=264, requires_grad=264, "
        "optimizer_listed=136, nonzero_gradient=None, actually_updated=None"
    ) in report
    assert "effective trainable DoF = 264" not in report
    assert "LN-LN 160 matched-to-136" not in report
    assert (
        "LN-LN optimizer-listed-count: 1 subunit, 160 parameters, "
        "nearest-feasible over-target optimizer-listed-count control"
    ) in report


def test_exact_136_lnln_claim_is_rejected_by_typed_error() -> None:
    # Given: the nearest feasible LN-LN active control has 160 parameters.
    lnln = profile_sets().active_dof.profile_named("LN-LN")

    # When / Then: claiming it exactly matches 136 optimizer-listed parameters is rejected.
    with pytest.raises(InvalidCapacityClaimError) as error:
        claim_exact_active_dof_match(lnln)
    assert error.value.model_name == "LN-LN"
    assert error.value.parameter_count == 160
    assert error.value.target_parameters == 136


def test_profiles_are_frozen_and_fail_closed_on_stale_source_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a frozen profile and a stale selector/count function.
    sets = profile_sets()
    with pytest.raises(FrozenInstanceError):
        setattr(sets.mechanistic, "total", 245)

    def stale_lnln_count(
        subunits_per_cell: int, cell_count: int, group_count: int
    ) -> int:
        if subunits_per_cell == 1:
            return 159
        return (
            subunits_per_cell * group_count * 16 + cell_count * subunits_per_cell + 80
        )

    monkeypatch.setattr(
        "evaluation.model_comparison.sample_efficiency_profiles.lnln_parameter_count",
        stale_lnln_count,
    )

    # When / Then: stale source values fail before a misleading profile is returned.
    with pytest.raises(ProfileSourceMismatchError) as error:
        profile_sets()
    assert error.value.model_name == "LN-LN"
    assert error.value.expected == 160
