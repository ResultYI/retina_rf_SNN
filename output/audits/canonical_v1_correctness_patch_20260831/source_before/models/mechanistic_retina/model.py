from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

import torch
from torch import nn

from models.mechanistic_retina.amacrine_pathways import AmacrinePathways
from models.mechanistic_retina.bipolar_subunits import (
    BipolarSubunits,
    PathFeatureBank,
    PathwaySpatialGeometry,
)
from models.mechanistic_retina.cell_specific_gains import build_cell_specific_gains
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticRetinaConfig,
    MechanisticRetinaOutput,
    PathwayClamp,
)
from models.mechanistic_retina.h1_pathway import H1Pathway
from models.mechanistic_retina.neural_operators import PathwayLocalOperator
from models.mechanistic_retina.pathway_gates import PathwayGates
from models.mechanistic_retina.rgc_state import RGCStateDynamics
from models.mechanistic_retina.shared_subunits import (
    SharedSubunitLayout,
    SharedSubunitMixer,
)
from models.mechanistic_retina.spatial_contract import register_spatial_contract
from models.mechanistic_retina.causal_contract import register_causal_contract
from models.mechanistic_retina.pathway_rf import pathway_basis_features, pathway_basis_rfs, pathway_base_rfs


@dataclass(frozen=True, slots=True)
class MechanisticModelError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


class MechanisticGraphTemporalRetina(nn.Module):
    def __init__(
        self,
        config: MechanisticRetinaConfig,
        cone_positions: torch.Tensor,
        cell_positions: torch.Tensor,
        cell_types: tuple[str, ...],
        polarities: tuple[str, ...],
        *,
        shared_subunit_edge_index: torch.Tensor | None = None,
        pathway_spatial_geometry: PathwaySpatialGeometry | None = None,
    ) -> None:
        super().__init__()
        cell_count = cell_positions.shape[0]
        if len(cell_types) != cell_count or len(polarities) != cell_count:
            raise MechanisticModelError("cell metadata lengths must match cell positions")
        invalid_types = tuple(sorted(set(cell_types) - {"midget", "parasol"}))
        if invalid_types:
            raise MechanisticModelError(
                f"unsupported cell types: {', '.join(invalid_types)}"
            )
        invalid_polarities = tuple(sorted(set(polarities) - {"ON", "OFF"}))
        if invalid_polarities:
            raise MechanisticModelError(
                f"unsupported polarities: {', '.join(invalid_polarities)}"
            )
        self.config = config
        mode = ArchitectureMode(config.architecture_mode)
        match mode:
            case ArchitectureMode.MECHANISM_IDENTIFIABLE:
                register_causal_contract(self)
                register_spatial_contract(self)
                mechanism_mode = True
                gate_initial = 0.5
                shared_radius = config.shared_subunit_radius_deg
            case ArchitectureMode.LEGACY:
                mechanism_mode = False
                gate_initial = 1.0
                shared_radius = 0.0
            case unreachable:
                assert_never(unreachable)
        self.h1 = H1Pathway(config, cone_positions)
        self.feature_bank = PathFeatureBank(
            config,
            cone_positions,
            cell_positions,
            cell_types,
            polarities,
            pathway_spatial_geometry,
        )
        self.shared_subunits = SharedSubunitMixer(
            SharedSubunitLayout(
                cell_positions,
                cell_types,
                polarities,
                shared_subunit_edge_index,
            ),
            radius_deg=shared_radius,
            trainable=mechanism_mode,
        )
        self.bipolar = BipolarSubunits(cell_types, polarities, shared=mechanism_mode)
        self.amacrine = AmacrinePathways(config, cell_types, polarities)
        self.gates = PathwayGates(
            gate_initial,
            group_index=self.amacrine.group_index,
            trainable=mechanism_mode,
            h1_amplitude=config.h1_amplitude,
            h1_amplitude_bounds=config.h1_amplitude_bounds,
        )
        self.cell_gains = build_cell_specific_gains(
            cell_count,
            config.cell_specific_gains,
            config.cell_specific_pathway_mixture,
        )
        self.operator = PathwayLocalOperator(config.operator_epsilon)
        self.rgc = RGCStateDynamics(config, cell_count)

    def forward(
        self,
        cones: torch.Tensor,
        *,
        observed_counts: torch.Tensor,
    ) -> MechanisticRetinaOutput:
        return self.forward_sequence(cones, observed_counts=observed_counts)

    def forward_sequence(
        self,
        cones: torch.Tensor,
        *,
        observed_counts: torch.Tensor,
        clamps: frozenset[PathwayClamp] = frozenset(),
        operators_enabled: bool = False,
    ) -> MechanisticRetinaOutput:
        if cones.ndim != 3 or observed_counts.ndim != 3:
            raise MechanisticModelError("inputs must be cone [B,T,C] and history [B,T,N]")
        clamps = frozenset(PathwayClamp(value) for value in clamps)
        if operators_enabled and ArchitectureMode(self.config.architecture_mode) is ArchitectureMode.MECHANISM_IDENTIFIABLE:
            raise MechanisticModelError("Canonical V1 uses only the shared BC encoder; independent pathway operators are disabled")
        gates = self.gates.values(clamps)
        h1 = self.h1(cones, amplitude=gates.h1)
        features = self.shared_subunits(self.feature_bank(h1.modulated_cones))
        modulation = self.operator(features, enabled=operators_enabled)
        bipolar = self.bipolar(
            features[:, :, :, :2], modulation[:, :, :, :2],
        )
        broad = self.bipolar(features[:, :, :, 2:], modulation[:, :, :, 2:])
        bc_direct = torch.stack((bipolar.sustained, bipolar.transient), dim=-1)
        bc_broad = torch.stack((broad.sustained, broad.transient), dim=-1)
        amacrine = self.amacrine(
            bc_broad,
            local_gate=gates.ac_local,
            transient_gate=gates.ac_transient,
        )
        direct_mask = bc_direct.new_tensor((
            PathwayClamp.DIRECT_BC_SUSTAINED not in clamps,
            PathwayClamp.DIRECT_BC_TRANSIENT not in clamps,
        ))
        pathway_currents = torch.cat(
            (bc_direct * direct_mask,
             torch.stack((amacrine.local_current, amacrine.transient_current), dim=-1)),
            dim=-1,
        )
        if self.cell_gains is not None:
            pathway_currents = self.cell_gains(pathway_currents)
        bc_sustained, bc_transient, ac_local, ac_transient = pathway_currents.unbind(-1)
        total = bc_sustained + bc_transient + ac_local + ac_transient
        rgc = self.rgc(
            total,
            observed_counts,
            adaptation_clamped=PathwayClamp.RGC_ADAPTATION in clamps,
            history_gate=gates.history,
        )
        on = self.bipolar.on_mask.view(1, 1, -1)
        off = self.bipolar.off_mask.view(1, 1, -1)
        return MechanisticRetinaOutput(
            h1.graph_drive,
            h1.state,
            h1.surround,
            bipolar.on_sustained,
            bc_sustained * on,
            bipolar.on_transient,
            bc_transient * on,
            bipolar.off_sustained,
            bc_sustained * off,
            bipolar.off_transient,
            bc_transient * off,
            bc_sustained,
            bc_transient,
            amacrine.local_state,
            ac_local,
            amacrine.transient_state,
            ac_transient,
            total,
            rgc.divisive,
            rgc.membrane,
            rgc.adaptation,
            rgc.history,
            rgc.logits,
            rgc.probability,
            bc_direct,
            bc_broad,
        )

    def pathway_basis_features(
        self,
        cones: torch.Tensor,
        *,
        clamps: frozenset[PathwayClamp] = frozenset(),
    ) -> torch.Tensor:
        return pathway_basis_features(self, cones, clamps)

    def pathway_basis_rfs(
        self, *, clamps: frozenset[PathwayClamp] = frozenset()
    ) -> torch.Tensor:
        return pathway_basis_rfs(self, clamps)

    def pathway_base_rfs(
        self, *, clamps: frozenset[PathwayClamp] = frozenset()
    ) -> tuple[torch.Tensor, ...]:
        return pathway_base_rfs(self, clamps)

    def project_mechanism_parameters(self) -> None:
        self.gates.project_()


def build_mechanistic_retina(
    config: MechanisticRetinaConfig,
    cone_positions: torch.Tensor,
    cell_positions: torch.Tensor,
    cell_types: tuple[str, ...],
    polarities: tuple[str, ...],
    *,
    shared_subunit_edge_index: torch.Tensor | None = None,
    pathway_spatial_geometry: PathwaySpatialGeometry | None = None,
) -> MechanisticGraphTemporalRetina:
    return MechanisticGraphTemporalRetina(
        config,
        cone_positions,
        cell_positions,
        cell_types,
        polarities,
        shared_subunit_edge_index=shared_subunit_edge_index,
        pathway_spatial_geometry=pathway_spatial_geometry,
    ).to(device=cone_positions.device, dtype=cone_positions.dtype)


__all__ = ["MechanisticGraphTemporalRetina", "MechanisticModelError", "build_mechanistic_retina"]
