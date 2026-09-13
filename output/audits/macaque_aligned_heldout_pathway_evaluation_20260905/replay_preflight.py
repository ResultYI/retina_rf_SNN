# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy", "opencv-python", "pydantic"]
# ///
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
INITIAL = json.loads((OUT / "evidence_manifest.initial.json").read_text())
USAGE = json.loads((OUT / "production_usage.json").read_text())
PRIMARY = ROOT / USAGE["aligned_root_resolved_from_analysis_manifest"]
ZERO = ROOT / INITIAL["lineages"]["zero_center_reference"]
LN = ROOT / INITIAL["lineages"]["LN"]
CNN = ROOT / INITIAL["lineages"]["CNN"]
sys.path.insert(0, str(PRIMARY))
sys.path.insert(0, str(ROOT))
from preflight import ADAPTER, MOVIE, alignment, factory, folder, load_data
from baselines.center_surround_ln import CenterSurroundLN
from data.schottdorf_lee_multirecording import load_schottdorf_movie_drive
from evaluation.mechanistic_retina.karamanlis_prediction_baselines import evaluate_retinal_model
from training.mechanistic_retina.center_surround_ln import evaluate_center_surround_ln
from training.mechanistic_retina.losses import expected_bernoulli_nll


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    assert not (OUT / "preflight.json").exists(), "No automatic replay restart"
    frozen = json.loads((OUT / "protocol_hash.json").read_text())
    assert sha(OUT / "TEST_PROTOCOL.md") == frozen["sha256"]
    assert json.loads((OUT / "data_usage_gate.json").read_text())["status"] == "CLEAN_FOR_CONFIRMATORY_EVALUATION"
    torch.set_num_threads(2)
    expected_runtime = json.loads((CNN / "results.json").read_text())["reporting_runtime"]["torch"]
    assert expected_runtime == json.loads((ZERO / "run-manifest.json").read_text())["torch_version"]
    assert str(torch.__version__) == expected_runtime
    rows = []
    result = dict(status="IN_PROGRESS", all_passed=False, cells=rows, protocol_sha256=frozen["sha256"],
        new_training_runs=0, heldout_predictions=0, heldout_NLL=0,
        start_utc=datetime.now(timezone.utc).isoformat(), runtime=dict(torch=str(torch.__version__), threads=2))

    def persist() -> None:
        (OUT / "preflight.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")

    persist()
    try:
        print("Checking frozen source, checkpoint and data identities", flush=True)
        for path, digest in INITIAL["input_sha256"].items():
            assert sha(ROOT / path) == digest, "Frozen input changed: " + path
        original = json.loads((ZERO / "run-manifest.json").read_text())["source_sha256"]
        original_data = {Path(k.replace("\\", "/")).name: v for k, v in original.items() if k.endswith(".mpg") or "lSS" in k}
        for path, digest in INITIAL["input_sha256"].items():
            if Path(path).name in original_data:
                assert digest == original_data[Path(path).name], "Production data identity changed"
        result["initial_inputs_verified"] = len(INITIAL["input_sha256"])
        source = {label: json.loads((root / "results.json").read_text())["cells"]
                  for label, root in (("zero", ZERO), ("aligned", PRIMARY), ("LN", LN), ("CNN", CNN))}
        lookup = {label: {c["cell_id"]: c for c in cells} for label, cells in source.items()}
        print("Decoding original development movie through the production loader", flush=True)
        movie = load_schottdorf_movie_drive(MOVIE, ADAPTER)
        for cell in source["zero"]:
            cid = cell["cell_id"]
            data = load_data(cid, movie)
            bundle = torch.load(CNN / "inputs" / (cid.replace("#", "_") + ".pt"), weights_only=True)
            for split_name in ("train", "validation"):
                actual = getattr(data, split_name)
                for key in ("cone_drive", "spike_counts", "spike_events", "valid_mask"):
                    assert torch.equal(getattr(actual, key), bundle[split_name][key]), (cid, split_name, key)
                assert tuple(bundle[split_name]["source_image_ids"]) == actual.source_image_ids
                assert tuple(bundle[split_name]["trial_indices"]) == actual.trial_indices
            for label, root, filename in (("zero", ZERO, "model-trained.pt"), ("aligned", PRIMARY, "model-trained.pt"), ("LN", LN, "ln-trained.pt")):
                cp_path = folder(cid, root) / filename
                cp = torch.load(cp_path, weights_only=True, map_location="cpu")
                assert cp["cell_id"] == cid
                if label in {"zero", "aligned"}:
                    assert cp["model_name"] == "Canonical V1" and cp["revision"] == 4 and cp["stage"] == "trained"
                    assert cp["model_config"]["causal_contract"] == "h1-shared-bc-direct-broad-ac"
                    assert cp["model_config"]["spatial_contract"] == "bc-central-disk_ac-overlapping-full-disk"
                    assert cp["steps"] == cp["training_contract"]["fresh_full_train_refit_steps"] == lookup[label][cid]["best_step"]
                    assert not cp["training_contract"]["original_validation_used_for_selection"]
                    assert torch.equal(cp["cell_positions_degs"], torch.zeros((1, 2)) if label == "zero" else alignment(cid))
                    assert torch.equal(cp["cone_positions_degs"], data.cone_positions_degs)
                    model = factory(cid, data, cp["cell_positions_degs"])
                    model.load_state_dict(cp["model"], strict=True)
                    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 33
                    metrics, logits = evaluate_retinal_model(model, data.validation)
                    expected = lookup[label][cid]["validation_nll_trained"]
                else:
                    assert cp["schema"] == "schottdorf_center_surround_separable_ln_v1"
                    assert cp["best_step"] == cp["refit_steps"] == lookup[label][cid]["refit_steps"]
                    model = CenterSurroundLN(cp["history"]["dt_ms"], cp["history"]["tau_ms"], cp["seed"])
                    model.load_state_dict(cp["model"], strict=True)
                    metrics, logits = evaluate_center_surround_ln(model, data.validation)
                    expected = lookup[label][cid]["ln_nll"]
                saved = torch.load(folder(cid, root) / "validation-predictions.pt", weights_only=True)
                checks = dict(strict_load=True,
                    target_exact=torch.equal(saved["target"], data.validation.spike_events),
                    mask_exact=torch.equal(saved["valid_mask"], data.validation.valid_mask),
                    source_order_exact=tuple(saved["source_image_ids"]) == data.validation.source_image_ids,
                    trial_order_exact=tuple(saved["trial_indices"]) == data.validation.trial_indices,
                    nll_exact=metrics.population_nll == expected,
                    checkpoint_state_unchanged=all(torch.equal(v, cp["model"][k]) for k, v in model.state_dict().items()),
                    no_gradients=all(p.grad is None for p in model.parameters()))
                if label in {"zero", "aligned"}:
                    checks["logits_bitwise_exact"] = torch.equal(logits, saved["logits_trained"])
                rows.append(dict(cell_id=cid, model=label, checks=checks, passed=all(checks.values()),
                    nll=metrics.population_nll, saved_nll=expected, delta_nll=metrics.population_nll - expected,
                    max_logit_error=float((logits - saved["logits_trained"]).abs().max()),
                    checkpoint_sha256=sha(cp_path), logits_sha256=tensor_sha(logits)))
                persist()
                if not all(checks.values()):
                    torch.save(dict(logits=logits, frozen_logits=saved["logits_trained"], target=saved["target"], mask=saved["valid_mask"]), OUT / "failed_development_replay.pt")
                    raise RuntimeError(f"STOP: exact frozen development replay failed: {label} {cid}")
                print(f"PASS {label} {cid}: exact development NLL", flush=True)
        print("Replaying CNN in its original frozen GPU environment", flush=True)
        subprocess.run(["D:/anaconda/envs/snn_env/python.exe", "-B", "-u", str(OUT / "replay_cnn_gpu.py")], check=True, cwd=ROOT)
        replay = torch.load(OUT / "cnn_development_replay.pt", weights_only=True)
        for cell in source["CNN"]:
            cid = cell["cell_id"]
            record = replay[cid]
            saved = torch.load(folder(cid, CNN) / "validation-predictions.pt", weights_only=True)
            value = float(expected_bernoulli_nll(record["logits"], saved["target"], saved["valid_mask"]))
            checks = dict(strict_load=record["strict_load"], checkpoint_state_unchanged=record["checkpoint_unchanged"], nll_exact=value == cell["cnn_nll"])
            rows.append(dict(cell_id=cid, model="CNN", checks=checks, passed=all(checks.values()),
                nll=value, saved_nll=cell["cnn_nll"], delta_nll=value-cell["cnn_nll"],
                gpu_logits_bitwise_exact=record["logits_bitwise_equal"],
                gpu_nll=record["gpu_nll"], saved_gpu_nll=record["saved_gpu_nll"],
                max_logit_error=float((record["logits"]-saved["logits_trained"]).abs().max()),
                checkpoint_sha256=sha(folder(cid, CNN) / "cnn-trained.pt"), logits_sha256=tensor_sha(record["logits"])))
            persist()
            assert all(checks.values()), f"STOP: CNN exact native CPU development NLL failed: {cid}"
            print(f"PASS CNN {cid}: exact original reporting NLL", flush=True)
        assert len(rows) == 88
        assert sha(OUT / "TEST_PROTOCOL.md") == frozen["sha256"]
        result.update(status="PASSED", all_passed=True, completed_utc=datetime.now(timezone.utc).isoformat())
        persist()
        print("ALL FOUR FROZEN FAMILIES PASSED; NO HELD-OUT INFERENCE YET", flush=True)
    except Exception as error:
        result.update(status="STOP_PREFLIGHT_FAILED", error=str(error), traceback=traceback.format_exc(),
            stopped_utc=datetime.now(timezone.utc).isoformat())
        persist()
        raise


if __name__ == "__main__":
    main()
