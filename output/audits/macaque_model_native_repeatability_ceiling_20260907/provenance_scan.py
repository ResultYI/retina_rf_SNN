# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# How to run: D:/anaconda/python.exe -B provenance_scan.py
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Final

OUT: Final = Path(__file__).resolve().parent
ROOT: Final = OUT.parents[2]
REPO: Final = ROOT / "data/real/schottdorf_lee_2021_repository"


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True, encoding="utf-8")


def main() -> None:
    assert not (OUT / "official_scan_evidence.json").exists()
    pattern = re.compile(r"corr|repeat|trial|2\s*ms|4\s*ms|0\.002|0\.004|lowpass|Table\s*2|Single Trials", re.I)
    files = git("ls-files", "-z").strip("\0").split("\0")
    entries = []
    hits = []
    for name in files:
        path = REPO / name
        payload = path.read_bytes()
        row = {"path": name, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload), "inspection": "binary_string_scan"}
        units = []
        if path.suffix == ".ipynb":
            cells = json.loads(payload)["cells"]
            units = [(f"cell_{i}_{cell['cell_type']}", "".join(cell.get("source", []))) for i, cell in enumerate(cells)]
            row.update(inspection="all_notebook_cells", code_cell_count=sum(c["cell_type"] == "code" for c in cells))
        elif zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                units = [(n, archive.read(n).decode("utf-8", errors="replace")) for n in archive.namelist() if n.endswith(".xml")]
            row["inspection"] = "all_archive_xml_members"
        else:
            units = [("file", payload.decode("utf-8", errors="replace"))]
        for label, text in units:
            matching = [{"line": i, "text": line[:2000]} for i, line in enumerate(text.splitlines(), 1) if pattern.search(line)]
            if matching:
                hits.append({"path": name, "unit": label, "matches": matching})
        entries.append(row)
    shallow = git("rev-parse", "--is-shallow-repository").strip() == "true"
    history = git("log", "--all", "--format=%H %s")
    deleted = git("log", "--all", "--diff-filter=D", "--name-status", "--format=%H")
    evidence = {"artifact_class": "NEW_DETERMINISTIC_DERIVED_ARTIFACT", "repository": str(REPO), "head": git("rev-parse", "HEAD").strip(), "tracked_file_count": len(entries), "shallow": shallow, "history_coverage": "LOCAL_AVAILABLE_ONLY", "available_history": history, "available_deleted_paths": deleted, "files": entries, "hits": hits}
    with (OUT / "official_scan_evidence.json").open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"files": len(entries), "hit_units": len(hits), "shallow": shallow, "notebooks": [x for x in entries if "code_cell_count" in x]}))


if __name__ == "__main__":
    main()
