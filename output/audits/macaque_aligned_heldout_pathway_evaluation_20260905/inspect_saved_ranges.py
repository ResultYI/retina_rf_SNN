from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SKIP = {".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache"}
RELEVANT = re.compile(r"schottdorf|macaque|compact_causal_cnn|real_data_independent_seed|spatial_contrast", re.I)
IDENTITY = re.compile(r"(?:lSS\d+-)?live-frames-(\d{6})-(\d{6})(?:-trial-(\d+))?")


def main() -> None:
    torch.set_num_threads(2)
    rows, ids, errors = [], [], []

    def inspect(name: str, payload: bytes) -> None:
        if not RELEVANT.search(name):
            return
        item = {"artifact_path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                "metadata_load": "NOT_NEEDED_MODEL_OR_RF_ARTIFACT", "source_ID_count": 0,
                "minimum_live_frame": "", "maximum_live_frame_inclusive": "", "fields": ""}
        if any(token in name.lower() for token in ("predict", "evaluation.pt", "inputs/", "input.pt", "replay.pt")):
            try:
                data = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
            except (RuntimeError, ValueError, EOFError, OSError, TypeError) as error:
                item["metadata_load"] = type(error).__name__
                errors.append({"path": name, "error": str(error)[:800]})
            else:
                ranges = []
                def visit(node: object, field: str) -> None:
                    if isinstance(node, dict):
                        for key, child in node.items():
                            if not isinstance(child, torch.Tensor):
                                visit(child, field + "." + str(key))
                    elif isinstance(node, (list, tuple)):
                        for index, child in enumerate(node):
                            if isinstance(child, (dict, list, tuple, str)):
                                visit(child, f"{field}[{index}]")
                    elif isinstance(node, str):
                        for match in IDENTITY.finditer(node):
                            start, stop = int(match[1]), int(match[2])
                            ranges.append((start, stop))
                            ids.append({"artifact_path": name, "field": field, "source_ID": match[0], "start": start, "stop_inclusive": stop})
                visit(data, "$")
                item.update(metadata_load="PASS_METADATA_ONLY", source_ID_count=len(ranges),
                            minimum_live_frame=min((r[0] for r in ranges), default=""),
                            maximum_live_frame_inclusive=max((r[1] for r in ranges), default=""),
                            fields=";".join(map(str, data)) if isinstance(data, dict) else type(data).__name__)
        rows.append(item)

    denied = []
    for current, dirs, files in os.walk(ROOT, onerror=denied.append):
        dirs[:] = [d for d in dirs if d not in SKIP and (Path(current) / d).resolve() != OUT]
        for name in files:
            path = Path(current) / name
            rel = path.relative_to(ROOT).as_posix()
            if path.suffix != ".pt" or not RELEVANT.search(rel):
                continue
            try:
                raw = path.read_bytes()
            except OSError as error:
                denied.append(error)
                continue
            inspect(rel, raw)
    for archive in (ROOT / ".local_archives/20260905-pre-audit-cleanup/recoverable-content.zip", ROOT / "output/retina_rf_SNN_code_review_20260831_122458.zip"):
        with zipfile.ZipFile(archive) as zf:
            for entry in zf.infolist():
                if entry.filename.endswith(".pt") and RELEVANT.search(entry.filename):
                    inspect(archive.relative_to(ROOT).as_posix() + "!" + entry.filename, zf.read(entry))
    tag = sys.argv[1]
    for filename, table in ((f"binary_usage_{tag}.csv", rows), (f"binary_source_ids_{tag}.csv", ids)):
        with (OUT / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    summary = {"artifacts": len(rows), "metadata_loaded": sum(r["metadata_load"] == "PASS_METADATA_ONLY" for r in rows),
        "source_ID_count": len(ids), "post20_source_ID_count": sum(r["stop_inclusive"] >= 3000 for r in ids),
        "errors": errors, "inaccessible": [str(error.filename) for error in denied], "new_model_outputs": 0}
    (OUT / f"binary_summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "inaccessible"}, indent=2))


if __name__ == "__main__":
    main()
