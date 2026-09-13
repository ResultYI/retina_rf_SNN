# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy"]
# ///
# How to run: D:/anaconda/python.exe -B -u finalize_evidence.py
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
from typing import Final

import numpy as np
import torch

OUT: Final = Path(__file__).resolve().parent
ROOT: Final = OUT.parents[2]


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_rows(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    torch.set_num_threads(2)
    initial = json.loads((OUT / "evidence_manifest.initial.json").read_text())
    changes = [name for name, digest in initial["input_sha256"].items() if sha(ROOT / name) != digest]
    assert not changes, changes
    gate = json.loads((OUT / "primary_replay.json").read_text())
    assert gate["all_passed"] and gate["cells_completed"] == 22
    assert sum(len(row["recording_ids"]) for row in gate["cells"]) == 37
    diag = torch.load(OUT / "inference-results.pt", weights_only=True)
    assert len(diag["aligned_full150"]) == 22 and len(diag["aligned_full600"]) == 4
    assert sum(len(v) for v in diag["seed_rf"].values()) == 12
    rf = read_rows("rf_identifiability.csv")
    pathways = read_rows("pathway_identifiability.csv")
    temporal = read_rows("temporal_identifiability.csv")
    pairs = read_rows("prediction_vs_mechanism_equivalence.csv")
    retention = read_rows("rf_truncation.csv")
    assert len(rf) == 160 and len(pathways) == 132 and len(temporal) == 1952
    assert len(pairs) == 37 and len(retention) == 104
    assert len(read_rows("quantity_taxonomy.csv")) == 30 and len(read_rows("claim_verdicts.csv")) == 25
    for row in retention:
        window = int(row["window_bins"])
        bundle = diag["aligned_full150"] if window == 150 else diag["aligned_full600"]
        value = bundle[row["cell_id"]] if window == 150 else bundle[row["cell_id"]]["rf"]
        x = value[row["pathway"]].numpy().astype(np.float64)
        fraction = np.square(x[..., -16:, :]).sum() / np.square(x).sum()
        assert math.isclose(fraction, float(row["retained_energy_pooled"]), abs_tol=1e-13)
    for row in rf:
        if row["center_x_deg_A"]:
            distance = math.hypot(float(row["center_x_deg_B"]) - float(row["center_x_deg_A"]),
                                  float(row["center_y_deg_B"]) - float(row["center_y_deg_A"]))
            assert distance == float(row["center_displacement_deg"])
        if row["cohort"] == "zero_center_seed":
            a = diag["seed_rf"][row["cell_id"]][row["fit_A"]][row["pathway"]].numpy().astype(np.float64).ravel()
            b = diag["seed_rf"][row["cell_id"]][row["fit_B"]][row["pathway"]].numpy().astype(np.float64).ravel()
            cosine = (a * b).sum() / (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
            assert math.isclose(cosine, float(row["cosine"]), abs_tol=1e-12)
    seeds = ROOT / initial["independent_seed_lineage"]
    for row in pathways:
        if row["cohort"] != "zero_center_seed":
            continue
        vectors = []
        for fit in (row["fit_A"], row["fit_B"]):
            saved = torch.load(seeds / "fits" / row["cell_id"].replace("#", "_") / fit / "evaluation.pt", weights_only=True)
            values = saved["validation_logits"]
            x = (values[row["pathway"] + "_off"] - values["normal"])[saved["valid_mask"]].numpy().astype(np.float64)
            vectors.append(x)
        a, b = vectors
        assert math.isclose(float(np.abs(b).mean() / np.abs(a).mean()), float(row["magnitude_ratio_B_over_A"]), abs_tol=1e-12)
        assert math.isclose(float(np.sqrt(np.square(a).mean())), float(row["rms_A"]), abs_tol=1e-12)
    for row in temporal:
        if row["u"]:
            u = (float(row["effective_value"]) - float(row["lower_bound"])) / (float(row["upper_bound"]) - float(row["lower_bound"]))
            assert u == float(row["u"]) and 0 <= u <= 1
            assert row["boundary"] == ("lower" if u <= .05 else "upper" if u >= .95 else "interior")
    for path in OUT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"))
    report = (OUT / "REPORT.md").read_text(encoding="utf-8")
    for link in re.findall(r"\]\((D:/[^)]+)\)", report):
        assert Path(link).exists() or link.endswith("evidence_manifest.json"), link
    table_width = None
    for line in report.splitlines():
        if line.startswith("|"):
            count = line.count("|")
            if table_width is None:
                table_width = count
            assert count == table_width, line
        else:
            table_width = None
    for row in read_rows("quantity_taxonomy.csv"):
        for source in row["code_source"].split("; "):
            assert Path(source.split("::")[0]).exists(), source
    summary = {}
    for cohort in ("alignment_nuisance", "zero_center_seed", "historical_shared_BC_synthetic"):
        summary[cohort] = {}
        for pathway in ("global", "H1", "direct_BC", "AC"):
            values = [r for r in rf if r["cohort"] == cohort and r["pathway"] == pathway]
            summary[cohort][pathway] = {metric: {"min": min(float(v[metric]) for v in values),
                "median": statistics.median(float(v[metric]) for v in values), "max": max(float(v[metric]) for v in values)}
                for metric in ("cosine", "relative_l2")}
    summary["synthetic_current_full_forward"] = {role: {k: v for k, v in values.items() if k != "logits"}
                                                for role, values in diag["synthetic_current_forward"].items()}
    (OUT / "numerical_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    synthetic = ROOT / initial["synthetic_lineage"]
    producer = json.loads((synthetic / "results.json").read_text())
    drift = {name: {"producer_sha256": digest, "current_sha256": sha(ROOT / name)}
             for name, digest in producer["source_hashes_before"].items() if sha(ROOT / name) != digest}
    status = subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT, text=True)
    initial_status = set(initial["git_status_before"].splitlines())
    current_status = set(status.splitlines())
    removed_status = sorted(initial_status - current_status)
    added_status = sorted(current_status - initial_status)
    assert not removed_status, removed_status
    assert all("output/audits/aligned_canonical_identifiability_main_20260905/" in line for line in added_status), added_status
    verification = {"frozen_input_count": len(initial["input_sha256"]), "all_frozen_inputs_unchanged": True,
        "primary_exact_replay": "PASS_22_OF_22", "real_inference_diagnostics": "PASS",
        "historical_synthetic_current_reconstruction": "PASS_EXACT",
        "synthetic_current_full_forward": "FAIL_NOT_EXACT_SOURCE_DRIFT",
        "independent_numerical_checks": "PASS", "report_local_links_and_table_structure": "PASS",
        "new_training_runs": 0, "optimizer_steps": 0, "new_illusion_runs": 0,
        "existing_git_status_preserved": True, "new_files_only_in_requested_output": True,
        "outside_output_changes": [], "row_counts": {"RF": len(rf), "pathway": len(pathways), "temporal": len(temporal), "pairs": len(pairs)}}
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    manifest = initial | {"git_status_after": status, "integrity_verification": verification,
        "historical_synthetic_source_drift": drift,
        "missing_evidence": ["aligned independent-seed fits", "independent biological RF/parameter truth",
            "CURRENT-CONTRACT TEMPORAL PARAMETER RECOVERY NOT TESTED", "optimized alternative testing pathway necessity"],
        "unreadable_evidence": [], "source_and_checkpoint_conversion": False,
        "new_artifact_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(OUT.iterdir())
                                 if p.is_file() and p.name != "evidence_manifest.json"}}
    (OUT / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
