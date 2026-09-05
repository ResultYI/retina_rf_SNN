from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from baselines.graph_tcn import graph_tcn_parameter_count, select_hidden_width
from baselines.lnln_subunit import lnln_parameter_count, select_subunit_count


CELL_COUNT: Final = 16
GROUP_COUNT: Final = 4
MECHANISTIC_TOTAL: Final = 264
MECHANISTIC_OPTIMIZER_LISTED: Final = 136
MECHANISTIC_REQUIRES_GRAD: Final = 264
BIAS_PARAMETERS: Final = 16
GLM_SH_PARAMETERS: Final = 7504


class FairnessRegime(StrEnum):
    ARCHITECTURE_SIZE = "architecture-size"
    ACTIVE_DOF = "optimizer-listed-count"


class ProfileRole(StrEnum):
    MECHANISTIC = "mechanistic"
    CAPACITY_CONTROL = "capacity-control"
    SHARED_UNMATCHED = "shared-unmatched"


class SelectorKind(StrEnum):
    NONE = "none"
    LN_LN_SUBUNITS = "lnln-subunits"
    GRAPH_TCN_WIDTH = "graph-tcn-width"


@dataclass(frozen=True, slots=True)
class ProfileSourceMismatchError(Exception):
    model_name: str
    observed: int
    expected: int

    def __str__(self) -> str:
        return (
            f"{self.model_name} profile source produced {self.observed}, "
            f"expected {self.expected}"
        )


@dataclass(frozen=True, slots=True)
class ProfileLookupError(Exception):
    model_name: str
    regime: FairnessRegime

    def __str__(self) -> str:
        return f"No {self.model_name} profile in {self.regime.value}"


@dataclass(frozen=True, slots=True)
class InvalidCapacityClaimError(Exception):
    model_name: str
    parameter_count: int
    target_parameters: int
    label: str

    def __str__(self) -> str:
        return (
            f"{self.model_name} has {self.parameter_count} parameters, "
            f"so it cannot be claimed as {self.target_parameters}: {self.label}"
        )


@dataclass(frozen=True, slots=True)
class MechanisticInventory:
    total: int
    requires_grad: int
    optimizer_listed: int
    nonzero_gradient: int | None
    actually_updated: int | None


@dataclass(frozen=True, slots=True)
class ModelProfile:
    model_name: str
    regime: FairnessRegime
    role: ProfileRole
    label: str
    selector_kind: SelectorKind
    selected_value: int | None
    parameter_count: int
    target_parameters: int | None


@dataclass(frozen=True, slots=True)
class CapacityClaim:
    model_name: str
    parameter_count: int
    target_parameters: int
    label: str


@dataclass(frozen=True, slots=True)
class FairnessProfileSet:
    regime: FairnessRegime
    profiles: tuple[ModelProfile, ...]

    def profile_named(self, model_name: str) -> ModelProfile:
        for profile in self.profiles:
            if profile.model_name == model_name:
                return profile
        raise ProfileLookupError(model_name, self.regime)


@dataclass(frozen=True, slots=True)
class FairnessProfileSets:
    mechanistic: MechanisticInventory
    architecture_size: FairnessProfileSet
    active_dof: FairnessProfileSet


def profile_sets() -> FairnessProfileSets:
    return _build_profile_sets()


def claim_exact_active_dof_match(profile: ModelProfile) -> CapacityClaim:
    if profile.parameter_count != MECHANISTIC_OPTIMIZER_LISTED:
        raise InvalidCapacityClaimError(
            profile.model_name,
            profile.parameter_count,
            MECHANISTIC_OPTIMIZER_LISTED,
            "exact optimizer-listed-count match",
        )
    return CapacityClaim(
        profile.model_name,
        profile.parameter_count,
        MECHANISTIC_OPTIMIZER_LISTED,
        "exact optimizer-listed-count match",
    )


def format_parameter_report(sets: FairnessProfileSets) -> str:
    lnln_active = sets.active_dof.profile_named("LN-LN")
    graph_active = sets.active_dof.profile_named("Graph-TCN")
    lnln_size = sets.architecture_size.profile_named("LN-LN")
    graph_size = sets.architecture_size.profile_named("Graph-TCN")
    return "\n".join(
        (
            "Mechanistic inventory: "
            f"total={sets.mechanistic.total}, "
            f"requires_grad={sets.mechanistic.requires_grad}, "
            f"optimizer_listed={sets.mechanistic.optimizer_listed}, "
            f"nonzero_gradient={sets.mechanistic.nonzero_gradient}, "
            f"actually_updated={sets.mechanistic.actually_updated}",
            "LN-LN architecture-size: "
            f"{lnln_size.selected_value} subunits, {lnln_size.parameter_count} parameters",
            "Graph-TCN architecture-size: "
            f"width {graph_size.selected_value}, {graph_size.parameter_count} parameters",
            "LN-LN optimizer-listed-count: "
            f"{lnln_active.selected_value} subunit, {lnln_active.parameter_count} "
            f"parameters, {lnln_active.label}",
            "Graph-TCN optimizer-listed-count: "
            f"width {graph_active.selected_value}, {graph_active.parameter_count} "
            f"parameters, {graph_active.label}",
            "Bias and GLM-SH: shared/unmatched reference baselines",
        )
    )


def _build_profile_sets() -> FairnessProfileSets:
    inventory = MechanisticInventory(
        total=MECHANISTIC_TOTAL,
        requires_grad=MECHANISTIC_REQUIRES_GRAD,
        optimizer_listed=MECHANISTIC_OPTIMIZER_LISTED,
        nonzero_gradient=None,
        actually_updated=None,
    )
    size_lnln = select_subunit_count(MECHANISTIC_TOTAL, CELL_COUNT, GROUP_COUNT)
    active_lnln = select_subunit_count(
        MECHANISTIC_OPTIMIZER_LISTED,
        CELL_COUNT,
        GROUP_COUNT,
    )
    size_graph = select_hidden_width(MECHANISTIC_TOTAL, CELL_COUNT)
    active_graph = select_hidden_width(MECHANISTIC_OPTIMIZER_LISTED, CELL_COUNT)
    _expect("LN-LN selector", size_lnln, 2)
    _expect("LN-LN active selector", active_lnln, 1)
    _expect("Graph-TCN selector", size_graph, 5)
    _expect("Graph-TCN active selector", active_graph, 2)
    size_lnln_parameters = lnln_parameter_count(size_lnln, CELL_COUNT, GROUP_COUNT)
    active_lnln_parameters = lnln_parameter_count(active_lnln, CELL_COUNT, GROUP_COUNT)
    size_graph_parameters = graph_tcn_parameter_count(size_graph, CELL_COUNT)
    active_graph_parameters = graph_tcn_parameter_count(active_graph, CELL_COUNT)
    _expect("LN-LN", size_lnln_parameters, 240)
    _expect("LN-LN", active_lnln_parameters, 160)
    _expect("Graph-TCN", size_graph_parameters, 270)
    _expect("Graph-TCN", active_graph_parameters, 144)
    return FairnessProfileSets(
        mechanistic=inventory,
        architecture_size=FairnessProfileSet(
            regime=FairnessRegime.ARCHITECTURE_SIZE,
            profiles=(
                _shared_profile(
                    "Bias", FairnessRegime.ARCHITECTURE_SIZE, BIAS_PARAMETERS
                ),
                _shared_profile(
                    "GLM-SH", FairnessRegime.ARCHITECTURE_SIZE, GLM_SH_PARAMETERS
                ),
                ModelProfile(
                    "LN-LN",
                    FairnessRegime.ARCHITECTURE_SIZE,
                    ProfileRole.CAPACITY_CONTROL,
                    "architecture-size nearest deterministic control",
                    SelectorKind.LN_LN_SUBUNITS,
                    size_lnln,
                    size_lnln_parameters,
                    MECHANISTIC_TOTAL,
                ),
                ModelProfile(
                    "Graph-TCN",
                    FairnessRegime.ARCHITECTURE_SIZE,
                    ProfileRole.CAPACITY_CONTROL,
                    "architecture-size nearest deterministic control",
                    SelectorKind.GRAPH_TCN_WIDTH,
                    size_graph,
                    size_graph_parameters,
                    MECHANISTIC_TOTAL,
                ),
                _mechanistic_profile(FairnessRegime.ARCHITECTURE_SIZE),
            ),
        ),
        active_dof=FairnessProfileSet(
            regime=FairnessRegime.ACTIVE_DOF,
            profiles=(
                _shared_profile("Bias", FairnessRegime.ACTIVE_DOF, BIAS_PARAMETERS),
                _shared_profile("GLM-SH", FairnessRegime.ACTIVE_DOF, GLM_SH_PARAMETERS),
                ModelProfile(
                    "LN-LN",
                    FairnessRegime.ACTIVE_DOF,
                    ProfileRole.CAPACITY_CONTROL,
                    "nearest-feasible over-target optimizer-listed-count control",
                    SelectorKind.LN_LN_SUBUNITS,
                    active_lnln,
                    active_lnln_parameters,
                    MECHANISTIC_OPTIMIZER_LISTED,
                ),
                ModelProfile(
                    "Graph-TCN",
                    FairnessRegime.ACTIVE_DOF,
                    ProfileRole.CAPACITY_CONTROL,
                    "near optimizer-listed-count matched",
                    SelectorKind.GRAPH_TCN_WIDTH,
                    active_graph,
                    active_graph_parameters,
                    MECHANISTIC_OPTIMIZER_LISTED,
                ),
                _mechanistic_profile(FairnessRegime.ACTIVE_DOF),
            ),
        ),
    )


def _shared_profile(
    model_name: str, regime: FairnessRegime, parameter_count: int
) -> ModelProfile:
    return ModelProfile(
        model_name,
        regime,
        ProfileRole.SHARED_UNMATCHED,
        "shared/unmatched reference baseline",
        SelectorKind.NONE,
        None,
        parameter_count,
        None,
    )


def _mechanistic_profile(regime: FairnessRegime) -> ModelProfile:
    return ModelProfile(
        "Mechanistic Retina",
        regime,
        ProfileRole.MECHANISTIC,
        "frozen mechanistic retina inventory",
        SelectorKind.NONE,
        None,
        MECHANISTIC_TOTAL,
        MECHANISTIC_OPTIMIZER_LISTED,
    )


def _expect(model_name: str, observed: int, expected: int) -> None:
    if observed != expected:
        raise ProfileSourceMismatchError(model_name, observed, expected)
