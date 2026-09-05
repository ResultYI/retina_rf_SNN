# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "matplotlib", "pydantic>=2"]
# ///
# How to run: D:/anaconda/python.exe -B -u output/real_data/schottdorf_r4_dev_circuit_replay_20260830/replay.py
# Use the frozen local runtime; do not resolve a different numerical environment.
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Final

import torch

ROOT: Final = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from data.retinal_recording import RealSequenceSplit
from evaluation.mechanistic_retina.clean_sampled_reporting import rf_bundle
from evaluation.mechanistic_retina.rf_effective import effective_rf
from evaluation.mechanistic_retina.schottdorf_fresh_evaluation import _CLAMPS
from evaluation.mechanistic_retina.schottdorf_ln_source import LNSourcePaths, load_ln_cell
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import sha256_file
from evaluation.mechanistic_retina.schottdorf_temporal_center_surround import _CLAMPS as PROBE_CLAMPS
from evaluation.mechanistic_retina.temporal_center_surround import CenterSurroundProbeConfig, build_center_surround_probe
from evaluation.mechanistic_retina.temporal_center_surround_reporting import (
    CellTrace, MetricRow, condition_metric_rows, group_metric_rows, write_metric_tables,
)
from evaluation.mechanistic_retina.temporal_center_surround_plots import save_figures
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina, build_mechanistic_retina

OLD: Final = ROOT / "output/real_data/schottdorf_lee_2021_22cell_canonical_v1_revision4_fresh_20260829"
NEW: Final = ROOT / "output/real_data/schottdorf_r4_development_22cell_20260830_verified"
OUT: Final = Path(__file__).resolve().parent
PROBE: Final = CenterSurroundProbeConfig()


@dataclass(frozen=True, slots=True)
class CircuitResult:
    rf: dict[str, torch.Tensor]
    perturbation: dict[str, dict[str, torch.Tensor]]
    normal: dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class ProbeResult:
    tensors: dict[str, torch.Tensor]
    rows: tuple[MetricRow, ...]
    trace: CellTrace


def circuit(model: MechanisticGraphTemporalRetina, split: RealSequenceSplit) -> CircuitResult:
    bundle = rf_bundle(model, split.cone_drive, split.spike_events)
    rf = {name: value.mean(dim=0) for name, value in bundle.items()}
    with torch.no_grad():
        normal = model.forward_sequence(split.cone_drive, observed_counts=split.spike_events)
    perturbations = {}
    zero_fields = {"H1_off": ("h1_surround_contribution",),
                   "BC_off": ("bc_sustained_current", "bc_transient_current"),
                   "AC_off": ("amacrine_local_current", "amacrine_transient_current")}
    for mode, clamps in _CLAMPS.items():
        with torch.no_grad():
            result = model.forward_sequence(split.cone_drive, observed_counts=split.spike_events, clamps=clamps)
        assert all(torch.count_nonzero(getattr(result, field)).item() == 0 for field in zero_fields[mode])
        clamped_rf = effective_rf(model, split.cone_drive, split.spike_events, clamps=clamps).mean(dim=0)
        perturbations[mode] = {"logit_delta": result.logits - normal.logits,
                               "probability_delta": result.spike_probability - normal.spike_probability,
                               "clamped_rf": clamped_rf, "rf_delta": clamped_rf - rf["global"]}
    return CircuitResult(rf, perturbations, {"logit": normal.logits, "probability": normal.spike_probability})


def temporal_probe(model: MechanisticGraphTemporalRetina, cell_id: str, group: str) -> ProbeResult:
    sign = 1.0 if group.endswith("ON") else -1.0
    probe = build_center_surround_probe(model.feature_bank.bc_support[0], model.feature_bank.ac_support[0],
                                         model.config.dt_ms, sign, PROBE)
    history = torch.zeros(probe.cone_drive.shape[0], probe.cone_drive.shape[1], 1)
    tensors, probabilities, rows = {}, {}, []
    for mode, clamps in PROBE_CLAMPS.items():
        with torch.no_grad():
            baseline = model.forward_sequence(torch.zeros_like(probe.cone_drive[:1]),
                                                observed_counts=torch.zeros_like(history[:1]), clamps=clamps)
            result = model.forward_sequence(probe.cone_drive, observed_counts=history, clamps=clamps)
        logit_delta = result.logits[..., 0] - baseline.logits[0, :, 0]
        probability_delta = result.spike_probability[..., 0] - baseline.spike_probability[0, :, 0]
        probabilities[mode] = probability_delta
        tensors.update({f"{mode}_logit": result.logits[..., 0], f"{mode}_probability": result.spike_probability[..., 0],
                        f"{mode}_logit_delta": logit_delta, f"{mode}_probability_delta": probability_delta})
        rows.extend(condition_metric_rows(cell_id, group, mode, probe, logit_delta, probability_delta, PROBE))
    tensors.update({"cone_drive": probe.cone_drive, "time_ms": probe.time_ms, "offset_ms": probe.offset_ms,
                    "center_support": model.feature_bank.bc_support[0], "surround_support": model.feature_bank.ac_support[0]})
    return ProbeResult(tensors, tuple(rows), CellTrace(cell_id, group, probe.time_ms, probe.names, probabilities))


def replay_error(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> dict[str, float | bool]:
    assert actual.keys() == expected.keys()
    errors, strict_passes = [], []
    for name, value in actual.items():
        assert value.shape == expected[name].shape
        assert torch.equal(torch.isnan(value), torch.isnan(expected[name]))
        assert not torch.isinf(value).any()
        strict_passes.append(torch.allclose(value, expected[name], rtol=1e-6, atol=1e-7, equal_nan=True))
        errors.append(float(torch.nan_to_num(value - expected[name]).abs().max()))
    return {"max_abs_error": max(errors), "strict_rtol1e6_atol1e7_pass": all(strict_passes)}


def main() -> None:
    assert not (OUT / "replay-results.json").exists()
    torch.set_num_threads(2)
    old_source = json.loads((OLD / "results.json").read_text())
    new_source = json.loads((NEW / "results.json").read_text())
    old_temporal = json.loads((OLD / "temporal_center_surround_perturbation/results.json").read_text())
    assert all(old_temporal["probe"][key] == value for key, value in asdict(PROBE).items())
    cells = old_source["cells"]
    assert [c["cell_id"] for c in cells] == [c["cell_id"] for c in new_source["cells"]]
    old_rf = torch.load(OLD / "evaluation/rf-tensors.pt", weights_only=True)
    old_perturbation = torch.load(OLD / "evaluation/perturbation-tensors.pt", weights_only=True)
    old_probe = torch.load(OLD / "temporal_center_surround_perturbation/response_tensors.pt", weights_only=True)
    paths = LNSourcePaths(ROOT / "data/real/schottdorf_lee_2021_repository",
                          ROOT / "data/real/schottdorf_lee_2021_macaque/1x10_256.mpg", OLD)
    code_paths = tuple((ROOT / "models/mechanistic_retina").glob("*.py")) + tuple(
        Path(module.__file__) for name, module in sys.modules.items()
        if name.startswith(("evaluation.mechanistic_retina", "data.schottdorf")) and getattr(module, "__file__", None))
    hashes = {str(path): sha256_file(path) for path in code_paths}
    for path in (OLD / "results.json", NEW / "results.json", OLD / "evaluation/rf-tensors.pt",
                 OLD / "evaluation/perturbation-tensors.pt", OLD / "temporal_center_surround_perturbation/response_tensors.pt"):
        hashes[str(path)] = sha256_file(path)
    output_rf, output_perturbation, normal_responses, probes = {}, {}, {}, {}
    old_replayed = {}
    rows, traces, checks = [], [], []
    for index, cell in enumerate(cells):
        cid, group = cell["cell_id"], f"{cell['retinal_class']}_{cell['polarity']}"
        loaded = load_ln_cell(paths, cid)
        data = loaded.data
        hashes.update(loaded.source_hashes)
        old_check = {}
        for label, source in (("old", OLD), ("R4-dev", NEW)):
            checkpoint_path = source / "cells" / cid.replace("#", "_") / "model-trained.pt"
            hashes[str(checkpoint_path)] = sha256_file(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            config = dict(checkpoint["model_config"])
            config["architecture_mode"] = ArchitectureMode(config["architecture_mode"])
            assert torch.equal(checkpoint["cone_positions_degs"], data.cone_positions_degs)
            assert torch.equal(checkpoint["cell_positions_degs"], data.cell_positions_degs)
            model = build_mechanistic_retina(MechanisticRetinaConfig(**config), data.cone_positions_degs,
                                              data.cell_positions_degs, data.cell_types, data.polarities)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            result = circuit(model, data.validation)
            probe_result = temporal_probe(model, cid, group)
            assert all(torch.equal(value, checkpoint["model"][name]) for name, value in model.state_dict().items())
            assert all(parameter.grad is None for parameter in model.parameters())
            if label == "old":
                old_check["rf_max_abs_error"] = replay_error(result.rf, old_rf[cid])
                old_check["perturbation_errors"] = {k: replay_error(result.perturbation[k], old_perturbation[cid][k]) for k in _CLAMPS}
                old_check["probe_max_abs_error"] = replay_error(probe_result.tensors, old_probe[cid])
                old_replayed[cid] = {"rf": result.rf, "perturbation": result.perturbation, "probe": probe_result.tensors}
            else:
                for key in ("cone_drive", "time_ms", "center_support", "surround_support"):
                    assert torch.equal(probe_result.tensors[key], old_probe[cid][key])
                prediction = torch.load(source / "cells" / cid.replace("#", "_") / "validation-predictions.pt", weights_only=True)
                assert torch.equal(prediction["target"], data.validation.spike_events)
                assert torch.equal(prediction["valid_mask"], data.validation.valid_mask)
                output_rf[cid], output_perturbation[cid] = result.rf, result.perturbation
                normal_responses[cid], probes[cid] = result.normal, probe_result.tensors
                rows.extend(probe_result.rows)
                traces.append(probe_result.trace)
        checks.append({"cell_id": cid, "old_artifact_replay": old_check, "model_and_checkpoint_unchanged": True,
                       "structural_clamps_exact_zero": True, "probe_inputs_match_old_exactly": True})
        print(f"REPLAYED {index+1}/22 {cid}: {old_check}", flush=True)
    for name, value in (("rf-tensors.pt", output_rf), ("perturbation-tensors.pt", output_perturbation),
                         ("validation-normal-responses.pt", normal_responses), ("response_tensors.pt", probes),
                         ("old-checkpoint-replayed.pt", old_replayed)):
        torch.save(value, OUT / name)
    write_metric_tables(OUT, rows, group_metric_rows(rows))
    save_figures(traces, OUT / "figures", PROBE)
    assert all(sha256_file(Path(path)) == digest for path, digest in hashes.items())
    payload = {"cell_count": len(cells), "checks": checks, "probe": old_temporal["probe"],
               "metric_contract": old_temporal["metric_contract"], "source_sha256": hashes,
               "execution": {"training_performed": False, "optimizer_created": False, "source_unchanged": True},
               "rf_lag_window_bins": 16, "rf_lag_window_ms": 16 * data.dt_ms,
               "rf_definition": "mean endpoint logit Jacobian over all validation sequences; last 16 input bins",
               "pathway_definition": "BC=RF(H1-off,AC-off); AC=RF(H1-off)-BC; H1=global-RF(H1-off)",
               "perturbation_aggregation": "all validation sequence bins, including warmup, exactly as old analysis",
               "old_artifact_bitwise_reproduction": "not assumed; strict comparisons and errors recorded; float-last-bit source UNVERIFIED",
               "rows": [asdict(row) for row in rows], "group_summary": group_metric_rows(rows)}
    (OUT / "replay-results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("ALL 22 REPLAYED; old tensor discrepancies recorded; no source/model/checkpoint changes", flush=True)


if __name__ == "__main__":
    main()
