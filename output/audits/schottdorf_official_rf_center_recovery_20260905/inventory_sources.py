# /// script
# requires-python = ">=3.11"
# dependencies = ["openpyxl", "pypdf"]
# ///
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile

from openpyxl import load_workbook
from pypdf import PdfReader

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
REPO = ROOT / "data/real/schottdorf_lee_2021_repository"
TERMS = [
    "RF center", "RF centre", "receptive field position", "x/y position",
    "screen position", "spot position", "stimulus displacement", "centering error",
    "cell position", "tangent screen", "DoG", "Gaussian", "fit center",
    "response map", "reverse corr", "RevCorr", "MxX", "MxY", "xoff", "yoff",
    "x0", "y0", "position", "center", "centre", "peak", "smoothing",
    "regulariz", "deconvol", "fft", "least", "eccentricity",
]
PATTERN = re.compile("|".join(re.escape(term) for term in TERMS), re.I)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    tracked = set(subprocess.check_output(
        ["git", "-C", str(REPO), "ls-files"], text=True
    ).splitlines())
    rows = []
    extracts = []
    examples = []
    notebook_centers = []
    for path in sorted(REPO.rglob("*")):
        if not path.is_file() or ".git" in path.relative_to(REPO).parts:
            continue
        rel = path.relative_to(REPO).as_posix()
        target = path
        status = "READ"
        kind = path.suffix
        text = ""
        note = ""
        if path.stat().st_size < 100 and path.read_bytes().startswith(b"/annex/objects/"):
            if path.name == "Fig2publication.xlsx":
                target = OUT / path.name
                note = "Downloaded official annex payload; MD5 matches pointer."
            else:
                status = "ANNEX_POINTER_ONLY"
                note = "Pointer inspected; payload not searched. Stimulus or model-visualization video, not coordinate table."
        if status == "READ":
            if kind == ".docx":
                with zipfile.ZipFile(target) as archive:
                    tree = ET.fromstring(archive.read("word/document.xml"))
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                text = "\n".join("".join(t.itertext()) for t in tree.findall(".//w:t", ns))
                note = "Complete document text; cell/recording catalog and acquisition notes."
            elif kind == ".ipynb":
                notebook = json.loads(target.read_text(encoding="utf-8"))
                parts = []
                for index, cell in enumerate(notebook["cells"]):
                    source = "".join(cell.get("source", []))
                    parts.append(f"CELL {index} {cell['cell_type']}\n{source}")
                    for output in cell.get("outputs", []):
                        parts.extend(["".join(output.get("text", [])),
                                      "".join(output.get("data", {}).get("text/plain", []))])
                    if "xoff" in source and "yoff" in source:
                        notebook_centers.append({"artifact": rel, "cell_index": index,
                            "source": source, "classification": "AUTHOR_PREDICTION_MODEL_PARAMETERS",
                            "eligible_independent_rc_center": False})
                text = "\n".join(parts)
                note = "All code/markdown cells and text outputs; embedded raster outputs not used to infer centers."
            elif kind == ".xlsx":
                workbook = load_workbook(target, read_only=True, data_only=True)
                parts = []
                for sheet in workbook:
                    for row in sheet:
                        for cell in row:
                            if isinstance(cell.value, str):
                                parts.append(f"{sheet.title}!{cell.coordinate}: {cell.value}")
                    if sheet["A1"].value == "MxFr":
                        for row_number in (2, 3):
                            examples.append({"source_artifact": rel, "sheet": sheet.title,
                                "source_fields": f"A{row_number}:D{row_number}",
                                "MxFr": sheet.cell(row_number, 1).value,
                                "MxX": sheet.cell(row_number, 2).value,
                                "MxY": sheet.cell(row_number, 3).value,
                                "Max": sheet.cell(row_number, 4).value,
                                "target_22_cell_match": False,
                                "method": "author saved reverse-correlation extrema; not 2D DoG fitted center",
                                "raw_unit": "map pixel index; index origin/orientation not verified"})
                workbook.close()
                with zipfile.ZipFile(target) as archive:
                    for name in archive.namelist():
                        if name.startswith("xl/charts/") and name.endswith(".xml"):
                            parts.append(name + "\n" + archive.read(name).decode("utf-8"))
                text = "\n".join(parts)
                note = "All sheet labels, hidden-sheet status, cached extrema and chart XML; formulas/defined names separately in workbook_inventory.json."
            elif kind == ".pdf":
                text = "\n".join(page.extract_text() or "" for page in PdfReader(target).pages)
                note = "Extracted every page: response-time model-fit plots, not RF position tables."
            elif kind in {".txt", ".csv", ".py", ".md", ".yml", ".sh", ".out"} or not kind:
                text = target.read_text(encoding="utf-8", errors="replace")
                note = "Complete text scanned."
            else:
                status = "BINARY_NOT_COORDINATE_TABLE"
                note = "Inventory only: gun-spectra image, video, or compiled duplicate of Python source."
        hits = [{"line": number, "text": line[:3000]}
                for number, line in enumerate(text.splitlines(), 1) if PATTERN.search(line)]
        if text:
            extracts.append({"artifact": rel, "hits": hits,
                             "all_text": text if kind in {".ipynb", ".docx", ".pdf"} else None})
        rows.append({"source_artifact": str(path.relative_to(ROOT)), "tracked": rel in tracked,
            "official_url": "https://gin.g-node.org/Manuel/Macaque-ganglion-cells/src/cffefb08c760f04c9a951da46061b361d2288e9b/" + rel,
            "bytes": path.stat().st_size, "sha256": sha(path),
            "payload_path": str(target.relative_to(ROOT)), "payload_sha256": sha(target),
            "kind": kind, "access_status": status, "searched_fields": " | ".join(TERMS),
            "matching_line_count": len(hits), "note": note})
    for filename, data in [("source_search_evidence.json", extracts),
                           ("official_saved_example_extrema.json", examples),
                           ("excluded_notebook_model_centers.json", notebook_centers)]:
        (OUT / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "source_inventory.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "search_summary.json").write_text(json.dumps({
        "repository_files": len(rows), "tracked_files": sum(row["tracked"] for row in rows),
        "extensions": sorted({row["kind"] for row in rows}), "searched_terms": TERMS,
        "saved_example_extrema_rows": len(examples),
        "author_rf_estimation_code_found": False,
        "target_22_usable_experiment_time_centers": 0,
        "target_22_usable_author_rc_centers": 0,
        "supplementary_statistical_summary": "UNVERIFIED_ACCESS_BLOCKED",
    }, indent=2), encoding="utf-8")
    print(f"Inventoried {len(rows)} files; extracted {len(examples)} saved example extrema rows.")


if __name__ == "__main__":
    main()
