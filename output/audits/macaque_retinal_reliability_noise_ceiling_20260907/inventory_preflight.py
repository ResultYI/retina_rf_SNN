from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[3]
OUT: Final = Path(__file__).resolve().parent
OFFICIAL: Final = ROOT / "data/real/schottdorf_lee_2021_repository"
ALIGNMENT: Final = ROOT / "output/audits/macaque_fixed_alignment_experiment_20260905"
PRIOR: Final = ROOT / "output/audits/macaque_aligned_heldout_pathway_evaluation_20260905"
PAPER: Final = ROOT / "output/audits/schottdorf_official_rf_center_recovery_20260905"
sys.path.insert(0, str(ROOT))

import numpy as np

from data.schottdorf_lee_catalog import RecordingKind, public_recordings
from data.schottdorf_lee_spikes import parse_recording_spike_trials


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    assert not (OUT / "inventory_lock.json").exists(), "Do not overwrite a completed inventory"
    alignment_inputs = json.loads((ALIGNMENT / "analysis-manifest.json").read_text())["inputs_sha256"]
    artifact_inputs = json.loads((ALIGNMENT / "artifact-manifest.json").read_text())["artifacts"]
    heldout = json.loads((PRIOR / "evidence_manifest.json").read_text())
    aligned = [ROOT / name for name in alignment_inputs if name.endswith("model-trained.pt")]
    assert len(aligned) == 22
    identities = []
    for path in aligned:
        key = str(path.relative_to(ROOT))
        assert sha(path) == alignment_inputs[key] == artifact_inputs[key]["sha256"]
        metadata = json.loads(path.with_name("results.json").read_text())
        identities.append({"cell_id": metadata["cell_id"], "model": "aligned", "path": relative(path), "sha256": sha(path)})
    comparators = [row for row in heldout["checkpoint_identity"] if row["model"] in {"LN", "CNN"}]
    assert len(comparators) == 44
    for item in comparators:
        assert sha(ROOT / item["path"]) == item["sha256"]
    identities += comparators
    cell_ids = [row["cell_id"] for row in identities if row["model"] == "aligned"]
    assert len(set(cell_ids)) == 22
    records = [r for r in public_recordings(OFFICIAL / "data") if r.cell_id in cell_ids]
    official_files = [p for p in OFFICIAL.rglob("*") if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and p.suffix in {".py", ".ipynb", ".md", ".txt", ".docx", ".yml"}]
    source_files = [ROOT / "data/schottdorf_lee_catalog.py", ROOT / "data/schottdorf_lee_spikes.py", ROOT / "data/schottdorf_lee_multirecording.py", ROOT / "data/schottdorf_lee_2021.py"]
    inputs = set(official_files + source_files + aligned)
    inputs.update(ROOT / row["path"] for row in comparators)
    inputs.update(path.with_name("results.json") for path in aligned)
    inputs.update([ALIGNMENT / "analysis-manifest.json", ALIGNMENT / "artifact-manifest.json", ALIGNMENT / "manifest.json", PRIOR / "evidence_manifest.json", PAPER / "paper.html", PAPER / "paper_bioc.xml", PAPER / "retrieval_log.json", ROOT / "data/real/schottdorf_lee_2021_macaque/1x10_256.mpg"])
    before = {relative(path): sha(path) for path in sorted(inputs)}
    inventory = []
    repeat_rows = []
    raw_times = {}
    for cell in cell_ids:
        cell_records = [record for record in records if record.cell_id == cell]
        repeats = [record for record in cell_records if record.recording_kind is RecordingKind.REPEATED_ONE_MINUTE]
        assert len(repeats) <= 1
        record = cell_records[0]
        trained = json.loads(next(path for path in aligned if path.parent.name == cell.replace("#", "_")).with_name("results.json").read_text())
        assert set(trained["recording_ids"]) == {r.recording_id for r in cell_records}
        assert trained["retinal_class"] == record.retinal_class and trained["polarity"] == record.polarity
        row = {"cell_id": cell, "group": record.retinal_class + "_" + record.polarity, "polarity": record.polarity, "recording_ids": ";".join(r.recording_id for r in cell_records), "repeated_recording_id": "", "repeat_count": 0, "duration_s": "", "spike_timing_resolution_ms": 0.1, "usable_for_reliability": False, "exclusion_reason": "NO_PUBLIC_RAW_6_REPEAT_RECORDING", "source": ";".join(relative(r.path) for r in cell_records), "source_sha256": ";".join(sha(r.path) for r in cell_records), "catalog_raw_kind_discrepancy": False, "repeated_record_used_by_current_training_contract": False, "reliability_computed": False}
        if repeats:
            repeat_record = repeats[0]
            payload = repeat_record.path.read_text(encoding="utf-8")
            assert "Video Starts\t" in payload and "Spikes per rpt" in payload and "Spikes times" in payload
            parsed = parse_recording_spike_trials(repeat_record)
            assert len(parsed.live_times_ms_by_trial) == 6 and parsed.resolution_ms == 0.1
            for index, times in enumerate(parsed.live_times_ms_by_trial):
                array = times.numpy()
                assert len(array) > 0 and np.all(np.isfinite(array)) and np.all(array[1:] >= array[:-1])
                key = cell.replace("#", "_") + f"_repeat_{index + 1}_time_ms"
                raw_times[key] = array.copy()
                repeat_rows.append({"cell_id": cell, "recording_id": repeat_record.recording_id, "repeat": index + 1, "duration_s": 60, "live_spike_count": len(array), "first_spike_ms": array[0], "last_spike_ms": array[-1], "video_start_ticks_internal_control": parsed.video_start_ticks, "duplicate_identical_payload_removed": parsed.duplicate_payload_removed, "raw_array_key": key, "source": relative(repeat_record.path)})
            row.update(repeated_recording_id=repeat_record.recording_id, repeat_count=6, duration_s=60, usable_for_reliability=True, exclusion_reason="", catalog_raw_kind_discrepancy=repeat_record.catalog_recording_kind is not repeat_record.recording_kind, repeated_record_used_by_current_training_contract=True)
        inventory.append(row)
    for name, rows in [("REPEATED_RECORDING_INVENTORY.csv", inventory), ("raw_repeat_metadata.csv", repeat_rows)]:
        with (OUT / name).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (OUT / "raw_repeated_spike_times.npz").open("xb") as stream:
        np.savez_compressed(stream, **raw_times)
    search = []
    pattern = re.compile(r"reliab|corrcoef|pearson|smooth|lowpass|low.pass", re.IGNORECASE)
    for path in official_files:
        if path.suffix == ".py":
            units = [(index, line) for index, line in enumerate(path.read_text().splitlines(), 1)]
        elif path.suffix == ".ipynb":
            notebook = json.loads(path.read_text())
            units = [(index, "".join(cell.get("source", []))) for index, cell in enumerate(notebook["cells"]) if cell["cell_type"] == "code"]
        else:
            continue
        search.append({"path": relative(path), "sha256": sha(path), "unit_kind": "line" if path.suffix == ".py" else "zero_based_notebook_code_cell", "matches": [{"unit": index, "text": text} for index, text in units if pattern.search(text)]})
    assert all(sha(ROOT / name) == digest for name, digest in before.items())
    group_count = Counter(row["group"] for row in inventory if row["usable_for_reliability"])
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "cohort_cells": len(inventory), "raw_repeat_eligible_cells": sum(row["usable_for_reliability"] for row in inventory), "eligible_groups": dict(group_count), "excluded_cells": [row["cell_id"] for row in inventory if not row["usable_for_reliability"]], "saved_live_repeats": len(repeat_rows), "official_repository_head": subprocess.check_output(["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True).strip(), "input_sha256": before, "input_hash_changes": [], "checkpoint_identity": identities, "model_checkpoints_loaded": 0, "model_inference": 0, "smoothing_performed": False, "correlations_computed": False, "training": 0, "parameter_fitting": 0, "new_seed": 0, "source_modifications": 0, "processing_code_sha256": sha(Path(__file__)), "output_sha256": {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}}
    for name, content in [("official_code_search.json", search), ("inventory_lock.json", summary)]:
        with (OUT / name).open("x", encoding="utf-8") as stream:
            json.dump(content, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: summary[key] for key in ["cohort_cells", "raw_repeat_eligible_cells", "eligible_groups", "excluded_cells", "saved_live_repeats", "model_inference", "smoothing_performed"]}))


if __name__ == "__main__":
    main()
