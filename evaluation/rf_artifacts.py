from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, TypeAlias

import torch

from evaluation.response_report_schema import ResponseReportEvidence
from evaluation.rf_history_contracts import (
    RFHistoryContract,
    RFHistoryContractError,
    RF_HISTORY_CONTRACTS,
    require_exact_history_contracts,
)
from training.response_data import PreparedResponseData

RF_ARTIFACT_SCHEMA: Final = "retina-rf-artifacts-v2"
RF_ARTIFACT_TOP_LEVEL_KEYS: Final = (
    "schema",
    "cell_ids",
    "cone_positions_degs",
    "lag_order",
    "conditional_static_by_history",
    "conditional_dynamic_by_history",
    "free_running",
)
RF_ARTIFACT_DYNAMIC_HISTORY_KEYS: Final = (
    "trained_low",
    "trained_high",
    "initialized_low",
    "initialized_high",
)
RF_ARTIFACT_STATIC_HISTORY_KEYS: Final = ("trained", "initialized")
RF_ARTIFACT_FREE_RUNNING_KEYS: Final = (
    "static_trained",
    "static_initialized",
    "dynamic_trained_low",
    "dynamic_trained_high",
    "dynamic_initialized_low",
    "dynamic_initialized_high",
)

RFKernelBlock: TypeAlias = dict[str, torch.Tensor]
RFHistoryKernelMap: TypeAlias = dict[RFHistoryContract, RFKernelBlock]


class RFArtifactError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RFKernelBlockSpec:
    label: str
    keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RFKernelContract:
    cell_count: int
    cone_count: int


@dataclass(frozen=True, slots=True)
class RFArtifact:
    cell_ids: tuple[str, ...]
    cone_positions_degs: tuple[tuple[float, float], ...]
    lag_order: str
    conditional_static_by_history: RFHistoryKernelMap
    conditional_dynamic_by_history: RFHistoryKernelMap
    free_running: RFKernelBlock


def write_rf_artifacts(
    output: Path,
    data: PreparedResponseData,
    evidence: ResponseReportEvidence,
) -> None:
    free_running = evidence.free_running_rf
    by_history = require_exact_history_contracts(evidence.conditional_rf_by_history)
    torch.save(
        {
            "schema": RF_ARTIFACT_SCHEMA,
            "cell_ids": data.cells.ids,
            "cone_positions_degs": torch.as_tensor(data.cone_positions_degs),
            "lag_order": "oldest_to_current",
            "conditional_static_by_history": {
                key: {
                    "trained": value.static_rf.kernels.detach().cpu(),
                    "initialized": (
                        value.initialized_static_rf.kernels.detach().cpu()
                    ),
                }
                for key, value in by_history.items()
            },
            "conditional_dynamic_by_history": {
                key: {
                    "trained_low": value.dynamic_rf.mean_low_kernel,
                    "trained_high": value.dynamic_rf.mean_high_kernel,
                    "initialized_low": value.initialized_dynamic_rf.mean_low_kernel,
                    "initialized_high": value.initialized_dynamic_rf.mean_high_kernel,
                }
                for key, value in by_history.items()
            },
            "free_running": {
                "static_trained": free_running.static_rf.kernels.detach().cpu(),
                "static_initialized": (
                    free_running.initialized_static_rf.kernels.detach().cpu()
                ),
                "dynamic_trained_low": free_running.dynamic_rf.mean_low_kernel,
                "dynamic_trained_high": free_running.dynamic_rf.mean_high_kernel,
                "dynamic_initialized_low": (
                    free_running.initialized_dynamic_rf.mean_low_kernel
                ),
                "dynamic_initialized_high": (
                    free_running.initialized_dynamic_rf.mean_high_kernel
                ),
            },
        },
        output / "rf_artifacts.pt",
    )


def load_rf_artifact(path: Path) -> RFArtifact:
    return validate_rf_artifact(
        torch.load(path, map_location="cpu", weights_only=True)
    )


def validate_rf_artifact(value) -> RFArtifact:
    if not isinstance(value, Mapping):
        raise RFArtifactError("rf artifact must be a mapping")
    if set(value) != set(RF_ARTIFACT_TOP_LEVEL_KEYS):
        raise RFArtifactError("rf artifact top-level keys must be exact")
    if value["schema"] != RF_ARTIFACT_SCHEMA:
        raise RFArtifactError("rf artifact schema must be retina-rf-artifacts-v2")
    cell_ids = _cell_ids(value["cell_ids"])
    cone_positions = _cone_positions(value["cone_positions_degs"])
    lag_order = _lag_order(value["lag_order"])
    contract = RFKernelContract(len(cell_ids), len(cone_positions))
    static = _history_kernels(
        value["conditional_static_by_history"],
        RFKernelBlockSpec("static history", RF_ARTIFACT_STATIC_HISTORY_KEYS),
        contract,
    )
    dynamic = _history_kernels(
        value["conditional_dynamic_by_history"],
        RFKernelBlockSpec("dynamic history", RF_ARTIFACT_DYNAMIC_HISTORY_KEYS),
        contract,
    )
    free_running = _kernel_block(
        value["free_running"],
        RFKernelBlockSpec("free_running", RF_ARTIFACT_FREE_RUNNING_KEYS),
        contract,
    )
    _require_matching_kernel_shapes(
        tuple(
            kernel
            for block in (*static.values(), *dynamic.values(), free_running)
            for kernel in block.values()
        )
    )
    return RFArtifact(
        cell_ids=cell_ids,
        cone_positions_degs=cone_positions,
        lag_order=lag_order,
        conditional_static_by_history=static,
        conditional_dynamic_by_history=dynamic,
        free_running=free_running,
    )


def _cell_ids(value) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise RFArtifactError("rf artifact cell_ids must be a string sequence")
    ids = tuple(value)
    if not ids or any(not isinstance(item, str) or item == "" for item in ids):
        raise RFArtifactError("rf artifact cell_ids must be a string sequence")
    return ids


def _cone_positions(value) -> tuple[tuple[float, float], ...]:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RFArtifactError(
            "rf artifact cone_positions_degs must be finite Nx2 coordinates"
        ) from exc
    if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] != 2:
        raise RFArtifactError(
            "rf artifact cone_positions_degs must be finite Nx2 coordinates"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise RFArtifactError(
            "rf artifact cone_positions_degs must be finite Nx2 coordinates"
        )
    return tuple((float(row[0]), float(row[1])) for row in tensor.tolist())


def _lag_order(value) -> str:
    if value != "oldest_to_current":
        raise RFArtifactError("rf artifact lag order must be oldest_to_current")
    return value


def _history_kernels(
    value,
    spec: RFKernelBlockSpec,
    contract: RFKernelContract,
) -> RFHistoryKernelMap:
    if not isinstance(value, Mapping):
        raise RFArtifactError(f"rf artifact {spec.label} must be exact")
    try:
        by_history = require_exact_history_contracts(value)
    except RFHistoryContractError as exc:
        raise RFArtifactError(f"rf artifact {spec.label} must be exact") from exc
    return {
        history: _kernel_block(by_history[history], spec, contract)
        for history in RF_HISTORY_CONTRACTS
    }


def _kernel_block(
    value,
    spec: RFKernelBlockSpec,
    contract: RFKernelContract,
) -> RFKernelBlock:
    if not isinstance(value, Mapping):
        raise RFArtifactError(f"rf artifact {spec.label} must be exact")
    if set(value) != set(spec.keys):
        raise RFArtifactError(f"rf artifact {spec.label} must be exact")
    return {
        key: _kernel_tensor(value[key], spec.label, contract)
        for key in spec.keys
    }


def _kernel_tensor(
    value,
    label: str,
    contract: RFKernelContract,
) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RFArtifactError(f"rf artifact {label} kernel must be tensor-like") from exc
    if (
        tensor.ndim != 3
        or tensor.shape[0] != contract.cell_count
        or tensor.shape[1] < 1
        or tensor.shape[2] != contract.cone_count
    ):
        raise RFArtifactError(
            "rf artifact kernel shape must be [cell,lag,cone] and match identity"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise RFArtifactError("rf artifact kernels must be finite")
    return tensor.detach().cpu()


def _require_matching_kernel_shapes(kernels: tuple[torch.Tensor, ...]) -> None:
    expected = kernels[0].shape
    if any(kernel.shape != expected for kernel in kernels[1:]):
        raise RFArtifactError("rf artifact kernel shape must match across artifact")


__all__ = [
    "RFArtifact",
    "RFArtifactError",
    "RF_ARTIFACT_DYNAMIC_HISTORY_KEYS",
    "RF_ARTIFACT_FREE_RUNNING_KEYS",
    "RF_ARTIFACT_SCHEMA",
    "RF_ARTIFACT_STATIC_HISTORY_KEYS",
    "RF_ARTIFACT_TOP_LEVEL_KEYS",
    "load_rf_artifact",
    "validate_rf_artifact",
    "write_rf_artifacts",
]
