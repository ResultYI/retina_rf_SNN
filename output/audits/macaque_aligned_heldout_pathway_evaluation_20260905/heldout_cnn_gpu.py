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


def main() -> None:
    assert not (OUT / "cnn_test_logits.pt").exists()
    assert json.loads((OUT / "preflight.json").read_text())["all_passed"]
    protocol = json.loads((OUT / "protocol_hash.json").read_text())["sha256"]
    assert hashlib.sha256((OUT / "TEST_PROTOCOL.md").read_bytes()).hexdigest() == protocol
    cnn = ROOT / ".omo/evidence/compact_causal_cnn_baseline"
    runtime = json.loads((cnn / "runtime.json").read_text())
    assert str(torch.__version__) == runtime["torch"] and torch.cuda.get_device_name() == runtime["gpu"]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    inputs = torch.load(OUT / "cnn_test_inputs.pt", weights_only=True)
    assert inputs["protocol_sha256"] == protocol
    output = {}
    for cid, record in inputs["cells"].items():
        cp = torch.load(cnn / "cells" / cid.replace("#", "_") / "cnn-trained.pt", weights_only=True)
        model = CompactCausalCNN(cp["history"]["dt_ms"], cp["history"]["tau_ms"], 61001).cuda()
        model.load_state_dict(cp["model"], strict=True)
        model.eval()
        drive = inputs["movie_sequences"][record["sequence_indices"]].cuda()
        events = record["events"].cuda()
        with torch.no_grad():
            history = model.history_feature(events).detach()
            logits = torch.cat([model.forward_with_history(drive[i:i+8], history[i:i+8])
                                for i in range(0, drive.shape[0], 8)]).cpu()
        assert torch.isfinite(logits).all()
        assert all(torch.equal(v.cpu(), cp["model"][k]) for k, v in model.state_dict().items())
        output[cid] = logits
        print(f"Held-out CNN inference {len(output)}/22 {cid}", flush=True)
    torch.save(dict(protocol_sha256=protocol, logits=output), OUT / "cnn_test_logits.pt")


if __name__ == "__main__":
    main()
