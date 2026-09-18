from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "work")]

from retipath_final_common import load_retipath
from test_retipath_mechanism_observation import check_causality, check_interventions, check_normal


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    output = ROOT / "output/evaluations/retipath_mechanism_observation_intervention_api_20260915"
    if (output / "intervention_correctness.json").exists():
        raise FileExistsError("Refusing to overwrite an existing correctness report")
    config_path = ROOT / "configs/retipath_final.json"
    config = json.loads(config_path.read_text())
    registry_path = ROOT / config["checkpoint_registry"]
    registry = json.loads(registry_path.read_text())
    protocol_path = ROOT / "output/experiments/retipath_spatial_ei_phase2_population/checkpoints/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protected_paths = [config_path, registry_path, protocol_path, *(
        ROOT / "models/mechanistic_retina" / (name + ".py") for name in (
            "retipath", "retipath_spatial_ei", "local_bc_nonlinearity", "contracts", "h1_pathway",
            "graph", "bipolar_subunits", "amacrine_pathways", "pathway_gates", "state", "cell_specific_gains",
            "pathway_temporal", "rgc_state", "shared_subunits",
        )
    )]
    protected_paths = [p for p in protected_paths if p.exists()]
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected_paths}
    start = time.monotonic()
    result = {
        "status": "PASS", "scope": "interface correctness only; no NLL/RF/illusion experiment",
        "model_identity": config, "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "environment": {"python": sys.version, "torch": torch.__version__, "device": "cpu", "dtype": "float32", "threads": 1},
        "training_updates": 0, "new_checkpoints": 0, "new_target_blocks": [],
        "data_access": "Existing train/development cache containers loaded; only first train sequence used; no scores or target analyses calculated",
        "normal": [], "interventions": [], "causality": [], "inputs": [],
    }
    representatives = {"69#4", "67#6", "68#4", "67#4"}
    checkpoints = []
    for cell, record in registry["cells"].items():
        ref = protocol["cells"][cell]
        input_path = Path(ref["input_path"])
        input_hash = sha(input_path)
        assert input_hash == ref["input_sha256"], cell
        cached = torch.load(input_path, map_location="cpu", weights_only=True)
        assert set(cached) == {"train", "development"}
        x = cached["train"]["cone_drive"][:1].clone()
        y = cached["train"]["spike_events"][:1].clone()
        result["inputs"].append({"cell": cell, "path": str(input_path), "sha256": input_hash,
                                  "sequence_index": 0, "split": "train", "shape": list(x.shape),
                                  "stimulus_sha256": tensor_sha(x), "history_sha256": tensor_sha(y)})
        for seed, checkpoint in record["RetiPath"].items():
            path = ROOT / checkpoint["path"]
            digest = sha(path)
            assert digest == checkpoint["sha256"], (cell, seed)
            checkpoints.append((path, digest))
            model, cp = load_retipath(path)
            assert str(cp["seed"]) == seed
            case = {"cell": cell, "group": record["group"], "seed": int(seed), "checkpoint": str(path.relative_to(ROOT)),
                    "checkpoint_sha256": digest, "selected_updates": checkpoint["selected_updates"]}
            result["normal"].append({**case, **check_normal(model, x, y)})
            if cell in representatives:
                result["interventions"].append({**case, **check_interventions(model, x, y)})
                if int(seed) == registry["seeds"][0]:
                    result["causality"].append({**case, "checks": check_causality(model, x, y)})
        print(f"Verified {cell}: frozen normal replay; " + ("interventions checked" if cell in representatives else "no extra experiment"), flush=True)
    after = {str(p.relative_to(ROOT)): sha(p) for p in protected_paths}
    assert before == after
    assert all(sha(p) == digest for p, digest in checkpoints)
    result["protected_source_sha256"] = before
    result["protected_sources_and_checkpoints_unchanged"] = True
    result["implementation_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in (
        ROOT / "evaluation/mechanistic_retina/mechanism_observation.py",
        ROOT / "tests/test_retipath_mechanism_observation.py", Path(__file__),
    )}
    result["summary"] = {"normal_cases": len(result["normal"]), "intervention_cases": len(result["interventions"]) * 6,
                         "causality_cases": len(result["causality"]) * 7, "normal_max_abs_error": 0.0,
                         "all_normal_logits_probability_bitwise_equal": True,
                         "physiologically_validated_variables": 0}
    result["limits"] = ["No GPU/backend portability claim", "No physiological validation or model-performance inference",
                         "Reparameterization was not implemented or tested; its algebraic plan is separate"]
    result["elapsed_seconds"] = time.monotonic() - start
    output.mkdir(parents=True, exist_ok=True)
    with (output / "intervention_correctness.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
