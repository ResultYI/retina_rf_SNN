from __future__ import annotations

import json
from pathlib import Path
from typing import assert_never

import torch


def write_grid(
    tmp_path: Path,
    *,
    seeds: tuple[int, ...] = (1, 2),
    budgets: tuple[int, ...] = (1, 4),
    shuffle_mode: str = "shuffled_type",
    matched_initialization: bool = False,
) -> list[Path]:
    paths = []
    for budget in budgets:
        for seed in seeds:
            for mode in ("type_aware", "type_blind", "cell_only", shuffle_mode):
                path = tmp_path / f"{mode}-s{seed}-b{budget}"
                write_run(
                    path,
                    mode=mode,
                    seed=seed,
                    budget=budget,
                    matched_initialization=matched_initialization,
                )
                paths.append(path)
    return paths


def write_run(
    path: Path,
    *,
    mode: str,
    seed: int,
    budget: int,
    matched_initialization: bool = False,
) -> None:
    path.mkdir()
    nll = {
        "type_aware": 0.8 if budget == 1 else 0.5,
        "type_blind": 1.0 if budget == 1 else 0.8,
        "cell_only": 0.9 if budget == 1 else 0.55,
        "shuffled_type": 1.05 if budget == 1 else 0.9,
        "balanced_shuffled_type": 1.05 if budget == 1 else 0.9,
    }[mode]
    if matched_initialization:
        nll = {
            "type_aware": 0.7,
            "type_blind": 0.85,
            "cell_only": 0.75,
            "balanced_shuffled_type": 0.8,
        }[mode]
    (path / "final_metrics.json").write_text(
        json.dumps(_metrics(nll, initial_nll=1.0 if matched_initialization else nll + 0.1)),
        encoding="utf-8",
    )
    (path / "run_manifest.json").write_text(
        json.dumps(_manifest(mode, seed, budget, matched_initialization)),
        encoding="utf-8",
    )
    (path / "run_status.json").write_text(
        json.dumps({"status": "COMPLETED", "evaluation_split": "validation"}),
        encoding="utf-8",
    )
    (path / "parameter_delta.json").write_text(
        json.dumps(
            [
                {
                    "name": "rgc.spatial_sigma.type_base_raw",
                    "delta_values": [nll, nll + seed / 100],
                }
            ]
        ),
        encoding="utf-8",
    )
    torch.save(_artifact(mode, seed, matched_initialization), path / "rf_artifacts.pt")


def mutate_manifest(run: Path, keys: tuple[str, ...], value: str | int) -> None:
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    current = manifest
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def mutate_artifact(
    run: Path,
    key: str,
    value: (
        str
        | int
        | tuple[str | int, ...]
        | dict[str, dict[str, torch.Tensor]]
        | torch.Tensor
    ),
) -> None:
    artifact = torch.load(run / "rf_artifacts.pt", map_location="cpu", weights_only=True)
    artifact[key] = value
    torch.save(artifact, run / "rf_artifacts.pt")


def remove_artifact_key(run: Path, key: str) -> None:
    artifact = torch.load(run / "rf_artifacts.pt", map_location="cpu", weights_only=True)
    del artifact[key]
    torch.save(artifact, run / "rf_artifacts.pt")


def mutate_artifact_history_kernel(
    run: Path,
    parent: str,
    history: str,
    key: str,
    value: torch.Tensor | None,
) -> None:
    artifact = torch.load(run / "rf_artifacts.pt", map_location="cpu", weights_only=True)
    block = artifact[parent][history]
    if value is None:
        del block[key]
    else:
        block[key] = value
    torch.save(artifact, run / "rf_artifacts.pt")


def mutate_artifact_free_kernel(
    run: Path,
    key: str,
    value: torch.Tensor | None,
) -> None:
    artifact = torch.load(run / "rf_artifacts.pt", map_location="cpu", weights_only=True)
    block = artifact["free_running"]
    if value is None:
        del block[key]
    else:
        block[key] = value
    torch.save(artifact, run / "rf_artifacts.pt")


def remove_history(run: Path, history: str) -> None:
    metrics = json.loads((run / "final_metrics.json").read_text(encoding="utf-8"))
    del metrics["dynamic_rf"]["by_history"][history]
    (run / "final_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def set_source_pairs(run: Path, pair_count: int) -> None:
    metrics = json.loads((run / "final_metrics.json").read_text(encoding="utf-8"))
    for history in metrics["dynamic_rf"]["by_history"].values():
        history["trained"]["pair_count"] = pair_count
    (run / "final_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def _metrics(nll: float, *, initial_nll: float):
    return {
        "evaluation_split": "validation",
        "response_prediction": {
            "conditional": {
                "nll": nll,
                "micro_bits_per_spike": 1.5 - nll,
                "macro_bits_per_spike": 1.4 - nll,
                "calibration_error": nll / 10,
                "per_cell_nll": [nll, nll + 0.1],
            },
            "initialized_conditional": {
                "nll": initial_nll,
                "micro_bits_per_spike": 1.5 - initial_nll,
                "macro_bits_per_spike": 1.4 - initial_nll,
                "calibration_error": initial_nll / 10,
                "per_cell_nll": [initial_nll, initial_nll + 0.1],
            },
            "glm_test": {"poison": "must_not_be_read"},
        },
        "dynamic_rf": {
            "by_history": {
                history: {
                    "trained": {
                        "pair_count": 3,
                        "teacher_primary_errors": [nll / 2, nll / 3],
                    }
                }
                for history in ("zero", "matched_observed", "standard_train_rate")
            }
        },
    }


def _manifest(mode: str, seed: int, budget: int, matched_initialization: bool):
    return {
        "config": {
            "seed": seed,
            "training": {"max_optimizer_steps": budget},
        },
        "dataset_fingerprint": "fingerprint",
        "evaluation_split": "validation",
        "parameter_sharing": {
            "mode": mode,
            "matched_initialization": matched_initialization,
            "shuffle_contract": (
                "within_polarity_balanced"
                if mode == "balanced_shuffled_type"
                else "none"
            ),
            "observed_type_labels": ["midget", "parasol"],
            "cell_polarities": [0, 1],
            "effective_type_labels": ["midget", "parasol"],
            "initial_effective_parameters": {"threshold": [0.2, 0.2]},
        },
    }


def _artifact(mode: str, seed: int, matched_initialization: bool):
    kernel = _kernel(mode, seed)
    initialized = (
        torch.tensor([[[0.5, 0.5]], [[0.5, 0.5]]])
        if matched_initialization
        else kernel
    )
    return {
        "schema": "retina-rf-artifacts-v2",
        "cell_ids": ("cell-a", "cell-b"),
        "cone_positions_degs": torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        "lag_order": "oldest_to_current",
        "conditional_static_by_history": {
            history: {"trained": kernel, "initialized": initialized}
            for history in ("zero", "matched_observed", "standard_train_rate")
        },
        "conditional_dynamic_by_history": {
            history: {
                "trained_low": kernel,
                "trained_high": kernel,
                "initialized_low": initialized,
                "initialized_high": initialized,
            }
            for history in ("zero", "matched_observed", "standard_train_rate")
        },
        "free_running": {
            "static_trained": kernel,
            "static_initialized": initialized,
            "dynamic_trained_low": kernel,
            "dynamic_trained_high": kernel,
            "dynamic_initialized_low": initialized,
            "dynamic_initialized_high": initialized,
        },
    }


def _kernel(mode: str, seed: int) -> torch.Tensor:
    match mode:
        case "type_aware":
            return torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        case "type_blind":
            value = 1.0 if seed == 1 else -1.0
            return torch.tensor([[[value, 0.0]], [[0.0, value]]])
        case "cell_only":
            value = 1.0 if seed == 1 else 0.5
            return torch.tensor([[[value, 0.0]], [[0.0, value]]])
        case "shuffled_type":
            if seed == 1:
                return torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]]])
            return torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        case "balanced_shuffled_type":
            if seed == 1:
                return torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]]])
            return torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        case unreachable:
            assert_never(unreachable)
