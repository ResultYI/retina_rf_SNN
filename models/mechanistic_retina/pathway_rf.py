from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from models.mechanistic_retina.contracts import PathwayClamp

if TYPE_CHECKING:
    from models.mechanistic_retina.model import MechanisticGraphTemporalRetina


def pathway_basis_features(
    model: MechanisticGraphTemporalRetina,
    cones: torch.Tensor,
    clamps: frozenset[PathwayClamp],
) -> torch.Tensor:
    """Linear BC basis terms; both views use the same external BC weight tensor."""
    clamps = frozenset(PathwayClamp(value) for value in clamps)
    gates = model.gates.values(clamps)
    modulated = model.h1(cones, amplitude=gates.h1).modulated_cones
    features = model.feature_bank(modulated, mixer=model.shared_subunits)
    broad = features[:, :, :, 2:]
    batch, time, cells, paths, spatial, temporal = broad.shape
    state_input = broad.permute(0, 1, 2, 4, 5, 3).reshape(batch, time, -1, paths)
    states = model.amacrine.presynaptic_states(state_input).reshape(
        batch, time, cells, spatial, temporal, paths
    ).permute(0, 1, 2, 5, 3, 4)
    ac_gates = -torch.stack((gates.ac_local, gates.ac_transient), dim=1)
    ac_basis = states * ac_gates.view(1, 1, cells, paths, 1, 1)
    direct_mask = features.new_tensor((
        PathwayClamp.DIRECT_BC_SUSTAINED not in clamps,
        PathwayClamp.DIRECT_BC_TRANSIENT not in clamps,
    )).view(1, 1, 1, 2, 1, 1)
    basis = torch.cat((features[:, :, :, :2] * direct_mask, ac_basis), dim=3)
    return model.cell_gains.scale_basis(basis) if model.cell_gains is not None else basis


def pathway_basis_rfs(
    model: MechanisticGraphTemporalRetina, clamps: frozenset[PathwayClamp]
) -> torch.Tensor:
    """Jacobian of the linear basis helper, in [N,4,S,R,lag,cone] order."""
    reference = model.feature_bank.spatial_basis
    lag_count = model.config.lag_steps
    cones = reference.new_zeros(1, lag_count, reference.shape[-1])
    with torch.enable_grad():
        jacobian = torch.autograd.functional.jacobian(
            lambda values: pathway_basis_features(model, values, clamps)[0, -1],
            cones,
        )
    return jacobian.squeeze(4)


def pathway_base_rfs(
    model: MechanisticGraphTemporalRetina, clamps: frozenset[PathwayClamp]
) -> tuple[torch.Tensor, ...]:
    """True forward current Jacobians; the lag window is not the full state memory."""
    reference = model.feature_bank.spatial_basis
    with torch.enable_grad():
        cones = reference.new_zeros(1, model.config.lag_steps, reference.shape[-1]).requires_grad_(True)
        history = reference.new_zeros(1, model.config.lag_steps, model.bipolar.group_index.numel())
        result = model.forward_sequence(cones, observed_counts=history, clamps=clamps)
        currents = (result.bc_sustained_current, result.bc_transient_current,
                    result.amacrine_local_current, result.amacrine_transient_current)
        kernels = []
        for current in currents:
            rows = [torch.autograd.grad(current[0, -1, cell], cones, retain_graph=True)[0][0]
                    for cell in range(current.shape[-1])]
            kernels.append(torch.stack(rows))
    return tuple(kernels)
