from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import torch
from torch import nn


CANONICAL_SPATIAL_CONTRACT: Final = "bc-central-disk_ac-overlapping-full-disk"
_STATE_KEY: Final = "_spatial_contract_id"


@dataclass(frozen=True, slots=True)
class SpatialContractError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message


def register_spatial_contract(model: nn.Module) -> None:
    model.register_buffer(
        _STATE_KEY,
        torch.tensor(list(CANONICAL_SPATIAL_CONTRACT.encode()), dtype=torch.uint8),
    )
    model.register_load_state_dict_pre_hook(_check_spatial_contract)


def _check_spatial_contract(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    prefix: str,
    _local_metadata: Mapping[str, int],
    _strict: bool,
    _missing_keys: list[str],
    _unexpected_keys: list[str],
    _error_msgs: list[str],
) -> None:
    identity = state_dict.get(prefix + _STATE_KEY)
    expected = module.get_buffer(_STATE_KEY)
    if (
        identity is None
        or identity.dtype != torch.uint8
        or not torch.equal(identity.cpu(), expected.cpu())
    ):
        raise SpatialContractError(
            f"Canonical V1 spatial contract mismatch: expected {CANONICAL_SPATIAL_CONTRACT}; "
            "legacy or unlabelled spatial checkpoints cannot be loaded"
        )
    _check_loaded_geometry(module, state_dict, prefix)


def _check_loaded_geometry(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    prefix: str,
) -> None:
    support_names = ("bc_support", "ac_support")
    supports = []
    for name in support_names:
        key = f"feature_bank.{name}"
        incoming = state_dict.get(prefix + key)
        expected = module.get_buffer(key)
        if incoming is None or not torch.equal(incoming.cpu(), expected.cpu()):
            raise SpatialContractError(
                f"Canonical V1 checkpoint spatial geometry must preserve full-disk {name}"
            )
        supports.append(incoming.cpu().bool())
    spatial = state_dict.get(prefix + "feature_bank.spatial_basis")
    paths = state_dict.get(prefix + "feature_bank.path_spatial_basis")
    if spatial is None or paths is None:
        raise SpatialContractError("Canonical V1 checkpoint spatial geometry is incomplete")
    if (
        spatial.shape != module.get_buffer("feature_bank.spatial_basis").shape
        or paths.shape != module.get_buffer("feature_bank.path_spatial_basis").shape
        or not bool(torch.isfinite(spatial).all() and torch.isfinite(paths).all())
        or bool((spatial < 0).any() or (paths < 0).any())
    ):
        raise SpatialContractError("Canonical V1 checkpoint spatial basis is invalid")
    bc, ac = supports
    expected_pattern = torch.stack((bc, bc, ac, ac), dim=1)[:, :, None]
    expected_pattern = expected_pattern & (spatial.cpu()[:, None] > 0)
    if (
        not torch.equal(paths.cpu() > 0, expected_pattern)
        or not torch.equal(paths[:, 0], paths[:, 1])
        or not torch.equal(paths[:, 2], paths[:, 3])
    ):
        raise SpatialContractError(
            "Canonical V1 checkpoint spatial basis violates shared full-disk geometry"
        )
