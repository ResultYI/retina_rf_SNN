from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import torch
from torch import nn


CANONICAL_CAUSAL_CONTRACT: Final = "h1-shared-bc-direct-broad-ac"
_STATE_KEY: Final = "_causal_contract_id"


class CausalContractError(RuntimeError):
    pass


def register_causal_contract(model: nn.Module) -> None:
    model.register_buffer(
        _STATE_KEY,
        torch.tensor(list(CANONICAL_CAUSAL_CONTRACT.encode()), dtype=torch.uint8),
    )
    model.register_load_state_dict_pre_hook(_check_causal_contract)


def _check_causal_contract(
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
    if identity is None or identity.dtype != torch.uint8 or not torch.equal(identity.cpu(), expected.cpu()):
        raise CausalContractError(
            f"Canonical V1 causal contract mismatch: expected {CANONICAL_CAUSAL_CONTRACT}; "
            "independent-AC-encoder or unlabelled checkpoints cannot be loaded"
        )
