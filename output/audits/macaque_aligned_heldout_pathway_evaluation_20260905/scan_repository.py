# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# How to run: D:/anaconda/python.exe -B scan_repository.py
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Final
import zipfile

ROOT: Final = Path(__file__).resolve().parents[3]
OUT: Final = Path(__file__).resolve().parent
TEXT_SUFFIXES: Final = {".py", ".json", ".jsonl", ".csv", ".tsv", ".md", ".txt", ".log", ".yaml", ".yml", ".toml", ".ipynb", ".ps1", ".sh", ".bat"}
SKIP: Final = {".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
IDENTITY: Final = re.compile(r"schottdorf|lSS\d{5}|1x10_256|6x1min|macaque", re.I)
RANGE: Final = re.compile(r"train_sequence_count|validation_sequence_count|sequence_steps|used_duration_s|used_live_frames|live-frames-\d{6}-\d{6}|frame_count|frame_start|frame_stop|start_frame|stop_frame|duration_s|window_start|window_stop|20.?60|3000|9000", re.I)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_csv(name: str, records: list[dict[str, str | int]]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def freeze() -> None:
    target = OUT / "evidence_manifest.initial.json"
    if target.exists():
        return
    prior = json.loads((ROOT / "output/paper/final_evidence_package_20260905/evidence_manifest.json").read_text(encoding="utf-8"))
    paths = {ROOT / p for p in prior["input_sha256"]}
    paths.update(ROOT / p for p in ("AGENTS.md", "AUDIT_INDEX.md", "audit/DATA_AVAILABILITY.md",
        "audit/cleanup_20260905/archived_files.csv", "audit/cleanup_20260905/backup_verification.json",
        ".local_archives/20260905-pre-audit-cleanup/recoverable-content.zip"))
    paths.update(p for folder in ("models", "training", "evaluation", "data", "baselines", "scripts")
                 for p in (ROOT / folder).rglob("*.py"))
    paths.update((ROOT / "data/real/schottdorf_lee_2021_repository/data").glob("lSS*.txt"))
    paths.add(ROOT / "data/real/schottdorf_lee_2021_macaque/1x10_256.mpg")
    state = {"timestamp_UTC": datetime.now(timezone.utc).isoformat(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status_at_audit_start": subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT, text=True),
        "lineages": prior["lineages"], "input_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)},
        "range_rules_sha256": digest(OUT / "RANGE_SELECTION_RULES.md"), "new_training_runs": 0, "new_model_outputs": 0}
    target.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main() -> None:
    freeze()
    census, hits, configs = [], [], []
    inaccessible = []

    def inspect(name: str, raw: bytes, origin: str) -> None:
        text = raw.decode("utf-8-sig", errors="replace")
        relevant = bool(IDENTITY.search(name) or IDENTITY.search(text))
        census.append({"artifact_path": name, "origin": origin, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                       "macaque_reference_found": str(relevant)})
        if not relevant:
            return
        for lineno, line in enumerate(text.splitlines(), 1):
            if RANGE.search(line):
                hits.append({"artifact_path": name, "line": lineno, "evidence": line[:1800]})
        if name.endswith(".json"):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                return
            def visit(node: object, field: str) -> None:
                if isinstance(node, dict):
                    if {"train_sequence_count", "validation_sequence_count", "sequence_steps"} <= node.keys():
                        train, val, steps = (int(node[k]) for k in ("train_sequence_count", "validation_sequence_count", "sequence_steps"))
                        configs.append({"artifact_path": name, "field": field, "train_sequences": train, "validation_sequences": val,
                            "sequence_bins": steps, "total_live_bins": (train + val) * steps,
                            "end_seconds_at_150Hz": (train + val) * steps / 150})
                    if "used_live_frames" in node:
                        configs.append({"artifact_path": name, "field": field + ".used_live_frames", "train_sequences": "",
                            "validation_sequences": "", "sequence_bins": "", "total_live_bins": node["used_live_frames"],
                            "end_seconds_at_150Hz": float(node["used_live_frames"]) / 150})
                    for key, child in node.items():
                        visit(child, field + "." + str(key))
                elif isinstance(node, list):
                    for index, child in enumerate(node):
                        if isinstance(child, (dict, list)):
                            visit(child, f"{field}[{index}]")
            visit(value, "$")

    for current, dirs, files in os.walk(ROOT, onerror=inaccessible.append):
        dirs[:] = [d for d in dirs if d not in SKIP and (Path(current) / d).resolve() != OUT]
        for name in files:
            path = Path(current) / name
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith("data/real/") and path.suffix.lower() in {".txt", ".csv", ".tsv"}:
                continue
            try:
                raw = path.read_bytes()
            except OSError as error:
                inaccessible.append(error)
                continue
            inspect(rel, raw, "filesystem")
    members = []
    for archive in (ROOT / ".local_archives/20260905-pre-audit-cleanup/recoverable-content.zip",
                    ROOT / "output/retina_rf_SNN_code_review_20260831_122458.zip"):
        with zipfile.ZipFile(archive) as zf:
            for item in zf.infolist():
                members.append({"archive": archive.relative_to(ROOT).as_posix(), "member": item.filename, "bytes": item.file_size, "CRC": item.CRC})
                if Path(item.filename).suffix.lower() in TEXT_SUFFIXES:
                    inspect(archive.relative_to(ROOT).as_posix() + "!" + item.filename, zf.read(item), "archive")
    write_csv("repository_search_inventory.csv", census)
    write_csv("data_usage_search_hits.csv", hits)
    write_csv("executed_adapter_configs.csv", configs)
    write_csv("archive_members.csv", members)
    summary = {"text_files_scanned": len(census), "macaque_referencing_files": sum(r["macaque_reference_found"] == "True" for r in census),
        "range_hit_count": len(hits), "adapter_config_count": len(configs), "archive_member_count": len(members),
        "post20_configs": [r for r in configs if float(r["end_seconds_at_150Hz"]) > 20],
        "scan_exclusions": sorted(SKIP), "raw_recording_text": "only file identity hashed; not included in scientific-text search",
        "inaccessible": [{"path": str(error.filename), "error": str(error)} for error in inaccessible],
        "source_only_not_execution": "A config or script hit alone does not establish actual training or evaluation; adjudication required",
        "new_model_inference": 0}
    (OUT / "search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
