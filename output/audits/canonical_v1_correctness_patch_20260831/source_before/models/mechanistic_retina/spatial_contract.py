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
