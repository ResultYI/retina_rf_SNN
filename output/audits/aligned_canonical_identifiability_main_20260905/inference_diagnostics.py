# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy", "pydantic", "opencv-python"]
# ///
# How to run: D:/anaconda/python.exe -B -u inference_diagnostics.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Final

import torch

OUT: Final = Path(__file__).resolve().parent
ROOT: Final = OUT.parents[2]
PRIMARY: Final = ROOT / "output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905"
ZERO: Final = ROOT / "output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830"
SEEDS: Final = ROOT / ".omo/evidence/real_data_independent_seed_sanity"
SYN: Final = ROOT / "output/synthetic_canonical_v1_shared_bc_noise_free_3seeds_20260830"
SELECTED: Final = ("67#4", "67#6", "68#4", "69#4")
sys.path.insert(0, str(ROOT))

from evaluation.mechanistic_retina.clean_sampled_reporting import explicit_delay_bounds, rf_bundle, tau_bounds
from evaluation.mechanistic_retina.noise_free_recovery_metrics import expected_metrics
from evaluation.mechanistic_retina.schottdorf_fresh_evaluation import learned_parameter_values
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig, PathwayClamp
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina, build_mechanistic_retina
from training.mechanistic_retina.losses import expected_bernoulli_nll


def load_model(path: Path, *, synthetic: bool = False) -> MechanisticGraphTemporalRetina:
    with torch.serialization.safe_globals([ArchitectureMode]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cone_key, cell_key, state_key = (("cone_positions", "cell_positions", "model_state") if synthetic
                                    else ("cone_positions_degs", "cell_positions_degs", "model"))
    model = build_mechanistic_retina(MechanisticRetinaConfig(**checkpoint["model_config"]),
        checkpoint[cone_key], checkpoint[cell_key], checkpoint["cell_types"], checkpoint["polarities"])
    model.load_state_dict(checkpoint[state_key], strict=True)
    assert model.config.causal_contract == "h1-shared-bc-direct-broad-ac"
    model.eval()
    return model


def full_rf(model: MechanisticGraphTemporalRetina, cones: torch.Tensor,
            events: torch.Tensor) -> dict[str, torch.Tensor]:
    gradients = []
    for clamps in (frozenset(), frozenset({PathwayClamp.H1}),
                   frozenset({PathwayClamp.H1, PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT})):
        stimulus = cones.detach().clone().requires_grad_(True)
        endpoint = model.forward_sequence(stimulus, observed_counts=events, clamps=clamps).logits[:, -1, 0]
        gradient = torch.autograd.grad(endpoint.sum(), stimulus)[0].detach()
        gradients.append(gradient[:, None])
    normal, h1_off, bc = gradients
    return {"global": normal, "H1": normal - h1_off, "direct_BC": bc, "AC": h1_off - bc}


def unchanged(model: MechanisticGraphTemporalRetina, state: dict[str, torch.Tensor]) -> None:
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def main() -> None:
    torch.set_num_threads(2)
    assert json.loads((OUT / "primary_replay.json").read_text())["all_passed"]
    assert not (OUT / "inference-results.pt").exists()
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), OUT / "PROTOCOL.md")}
    (OUT / "inference-source-hashes.json").write_text(json.dumps(hashes, indent=2))
    validation = torch.load(OUT / "validation-replay.pt", weights_only=True)
    saved_rf = torch.load(PRIMARY / "rf-tensors.pt", weights_only=True)
    result = {"aligned_full150": {}, "aligned_full600": {}, "seed_rf": {}, "parameters": {},
              "bounds": {}, "bc_temporal_basis": {}, "synthetic_logits": {}, "synthetic_metrics": {}}
    checks = []
    for cid, data in validation.items():
        path = PRIMARY / "cells" / cid.replace("#", "_") / "model-trained.pt"
        model = load_model(path)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        rf = full_rf(model, data["cones"], data["events"])
        errors = {name: float((values[..., -16:, :].mean(0) - saved_rf[cid]["aligned"][name]).abs().max())
                  for name, values in rf.items()}
        assert all(error == 0 for error in errors.values()), (cid, errors)
        result["aligned_full150"][cid] = rf
        result["bounds"][cid] = ({f"tau_{k}": v.clone() for k, v in tau_bounds(model).items()}
                                  | {f"delay_{k}": v.clone() for k, v in explicit_delay_bounds(model).items()})
        result["bc_temporal_basis"][cid] = model.feature_bank.temporal_basis.detach().clone()
        if cid in SELECTED:
            first_trial = data["trial_indices"][0]
            recording = data["source_image_ids"][0].split("-live-")[0]
            selected = [i for i, (source, trial) in enumerate(zip(data["source_image_ids"], data["trial_indices"], strict=True))
                        if trial == first_trial and source.startswith(recording + "-live-")]
            assert len(selected) == 4
            ranges = [re.search(r"frames-(\d+)-(\d+)", data["source_image_ids"][i]) for i in selected]
            assert all(match is not None for match in ranges)
            starts = [int(match[1]) for match in ranges if match is not None]
            assert all(b - a == 150 for a, b in zip(starts, starts[1:]))
            cones = data["cones"][selected].reshape(1, 600, -1)
            events = data["events"][selected].reshape(1, 600, 1)
            result["aligned_full600"][cid] = {"rf": full_rf(model, cones, events),
                "cones": cones, "events": events, "source_indices": selected,
                "source_image_ids": [data["source_image_ids"][i] for i in selected]}
        unchanged(model, state)
        checks.append({"cell": cid, "kind": "aligned", "rf_max_error": errors, "state_unchanged": True})
        print(f"RF original 150 bins / fixed long diagnostic: {cid}", flush=True)
    for cid in SELECTED:
        result["seed_rf"][cid], result["parameters"][cid] = {}, {}
        data = validation[cid]
        for label in ("primary", "fresh_1", "fresh_2"):
            folder = SEEDS / "fits" / cid.replace("#", "_") / label
            evaluation = torch.load(folder / "evaluation.pt", weights_only=True)
            assert torch.equal(evaluation["target"], data["events"])
            assert torch.equal(evaluation["valid_mask"], data["mask"])
            assert tuple(evaluation["source_image_ids"]) == tuple(data["source_image_ids"])
            assert tuple(evaluation["trial_indices"]) == tuple(data["trial_indices"])
            path = ((ZERO / "cells" / cid.replace("#", "_") / "model-trained.pt") if label == "primary"
                    else folder / "model-trained.pt")
            model = load_model(path)
            state = {k: v.clone() for k, v in model.state_dict().items()}
            bundle = rf_bundle(model, data["cones"], data["events"])
            rf = {name: bundle[key].detach().mean(0) for name, key in
                  (("global", "global"), ("H1", "H1"), ("direct_BC", "BC"), ("AC", "AC"))}
            if label == "primary":
                assert all(torch.equal(value, saved_rf[cid]["zero"][name]) for name, value in rf.items())
            result["seed_rf"][cid][label] = rf
            result["parameters"][cid][label] = dict(learned_parameter_values(model))
            unchanged(model, state)
            checks.append({"cell": cid, "kind": label, "strict_load": True, "data_identity": True, "state_unchanged": True})
            print(f"Existing zero-center seed RF: {cid} {label}", flush=True)
    stimulus = torch.load(SYN / "stimulus-and-probabilities.pt", weights_only=True)
    source = json.loads((SYN / "results.json").read_text())
    assert source["parameter_recovery_audit_run"] is False
    archived = torch.load(SYN / "causal-replay-tensors.pt", weights_only=True)
    counterfactual = torch.load(SYN / "counterfactual-tensors.pt", weights_only=True)
    result["synthetic_current_forward"] = {}
    for name in ("rgc_state.py", "state.py"):
        key = next(k for k in source["source_hashes_before"] if k.replace("\\", "/") == "models/mechanistic_retina/" + name)
        assert hashlib.sha256((ROOT / key).read_bytes()).hexdigest() == source["source_hashes_before"][key]
    for role in ("teacher", "53001", "53002", "53003"):
        filename = "teacher.pt" if role == "teacher" else f"student-seed-{role}-trained.pt"
        model = load_model(SYN / filename, synthetic=True)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        target = stimulus["validation_teacher_probability"]
        with torch.no_grad():
            output = model.forward_sequence(stimulus["validation_cones"], observed_counts=torch.zeros_like(target))
        metrics_current = expected_metrics(model, stimulus["validation_cones"], target)
        reconstructed = {}
        for mode, currents in archived[role].items():
            total = (currents["bc_sustained_current"] + currents["bc_transient_current"]
                     + currents["amacrine_local_current"] + currents["amacrine_transient_current"])
            with torch.no_grad():
                reconstructed[mode] = model.rgc(total, torch.zeros_like(target), adaptation_clamped=False,
                                                history_gate=model.gates.values(frozenset()).history)
        historical = reconstructed["normal"]
        for mode in ("H1_off", "direct_BC_off", "AC_off"):
            assert torch.equal(reconstructed[mode].logits - historical.logits, counterfactual[role][mode]["logit_delta"])
        ce = expected_bernoulli_nll(historical.logits, target, torch.ones_like(target))
        entropy = expected_bernoulli_nll(torch.logit(target.clamp(1e-7, 1 - 1e-7)), target, torch.ones_like(target))
        metrics = {"expected_ce": float(ce), "kl": float(ce - entropy)}
        if role == "teacher":
            assert torch.equal(historical.probability, target)
        else:
            assert metrics["kl"] == source["solutions"][role]["validation_kl"]["trained"]
        result["synthetic_current_forward"][role] = {"logits": output.logits, "metrics": metrics_current,
            "max_probability_error_vs_historical": float((output.spike_probability - historical.probability).abs().max()),
            "historical_probability_exact": torch.equal(output.spike_probability, historical.probability),
            "h1_state_exact": torch.equal(output.h1_state, archived[role]["normal"]["h1_state"]),
            "bc_direct_max_error": float((output.bc_direct_presynaptic - archived[role]["normal"]["bc_direct_presynaptic"]).abs().max())}
        result["synthetic_logits"][role] = historical.logits
        result["synthetic_metrics"][role] = metrics
        unchanged(model, state)
    torch.save(result, OUT / "inference-results.pt")
    (OUT / "inference-verification.json").write_text(json.dumps({"checks": checks, "all_passed": True,
        "source_sha256": hashes, "training_runs": 0, "synthetic_current_full_forward_exact": False,
        "synthetic_archived_current_reconstruction_exact": True}, indent=2))
    print("Inference-only diagnostics completed; all identity checks passed", flush=True)


if __name__ == "__main__":
    main()
