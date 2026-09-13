# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.10.0", "numpy==1.26.4"]
# ///
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
from baselines.compact_causal_cnn import CompactCausalCNN
from data.retinal_recording import RealSequenceSplit
from training.mechanistic_retina.compact_causal_cnn import evaluate_cnn


def main() -> None:
    assert not (OUT / "cnn_development_replay.pt").exists()
    frozen = json.loads((OUT / "protocol_hash.json").read_text())
    assert hashlib.sha256((OUT / "TEST_PROTOCOL.md").read_bytes()).hexdigest() == frozen["sha256"]
    assert torch.cuda.is_available()
    cnn = ROOT / ".omo/evidence/compact_causal_cnn_baseline"
    runtime = json.loads((cnn / "runtime.json").read_text())
    assert str(torch.__version__) == runtime["torch"]
    assert torch.cuda.get_device_name() == runtime["gpu"]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    result = {}
    for cell in json.loads((cnn / "results.json").read_text())["cells"]:
        cid = cell["cell_id"]
        folder = cnn / "cells" / cid.replace("#", "_")
        bundle = torch.load(cnn / "inputs" / (cid.replace("#", "_") + ".pt"), weights_only=True)
        split = RealSequenceSplit(**bundle["validation"])
        gpu = RealSequenceSplit(split.cone_drive.cuda(), split.spike_counts.cuda(), split.spike_events.cuda(),
            split.valid_mask.cuda(), split.source_image_ids, split.trial_indices)
        cp = torch.load(folder / "cnn-trained.pt", weights_only=True)
        assert cp["schema"] == "schottdorf_compact_causal_cnn_v1" and cp["cell_id"] == cid
        assert cp["best_step"] == cp["refit_steps"] == cell["refit_steps"]
        model = CompactCausalCNN(cp["history"]["dt_ms"], cp["history"]["tau_ms"], 61001).cuda()
        model.load_state_dict(cp["model"], strict=True)
        metrics, logits = evaluate_cnn(model, gpu)
        saved = torch.load(folder / "validation-predictions.pt", weights_only=True)
        fit = json.loads((folder / "results.json").read_text())
        state_equal = all(torch.equal(v.cpu(), cp["model"][k]) for k, v in model.state_dict().items())
        result[cid] = dict(logits=logits.cpu(), strict_load=True, checkpoint_unchanged=state_equal,
            logits_bitwise_equal=torch.equal(saved["logits_trained"], logits.cpu()),
            gpu_nll=metrics.population_nll, saved_gpu_nll=fit["validation_nll_trained_gpu"])
        torch.save(result, OUT / "cnn_development_replay.pt")
        assert state_equal
        print(f"CNN GPU replay {len(result)}/22 {cid}", flush=True)
    (OUT / "cnn_development_runtime.json").write_text(json.dumps(dict(torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name(), cells=len(result), dtype="float32", amp=False, tf32=False,
        deterministic_algorithms=True, cudnn_benchmark=False), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
