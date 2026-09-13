import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
FIXED = ROOT / "output/audits/macaque_fixed_alignment_experiment_20260905"


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ranks(values: list[float]) -> list[float]:
    return [sum(other < value for other in values) + (sum(other == value for other in values) + 1) / 2
            for value in values]


def main() -> None:
    frozen = json.loads((OUT / "evidence_manifest.json").read_text())
    with (FIXED / "paired_prediction.csv").open(encoding="utf-8-sig", newline="") as handle:
        paired = {row["cell_id"]: row for row in csv.DictReader(handle)}
    old_summary = json.loads((FIXED / "summary.json").read_text())
    rows = []
    relations = []
    for cell in frozen["cells"]:
        row = paired[cell["cell_id"]]
        xp, yp = cell["LN_center_pixels"]
        assert float(row["ln_x_pixels"]) == xp and float(row["ln_y_pixels"]) == yp
        x, y = float(row["fixed_x_degrees"]), float(row["fixed_y_degrees"])
        assert abs(x - xp * 3 * 4.6 / 256) < 3e-8
        assert abs(y + yp * 3 * 4.6 / 256) < 3e-8
        offset = float(row["radial_offset_degrees"])
        assert abs(offset - math.hypot(x, y)) < 3e-8
        common = {"cell_id": cell["cell_id"], "recording_ids": "|".join(cell["recording_ids"]),
            "retinal_class": cell["retinal_class"], "polarity": cell["polarity"],
            "ln_x_degree": x, "ln_y_degree": y, "ln_radial_degree": offset,
            "ln_radial_arcmin": offset * 60, "official_x_degree": None,
            "official_y_degree": None, "official_radial_degree": None,
            "official_radial_arcmin": None, "vector_error_degree": None}
        rows.append({**common, "x_sign_agreement": None, "y_sign_agreement": None,
            "cosine": None, "angle_degree": None, "quadrant_agreement": None,
            "official_source_artifact": None, "official_evidence_class": None,
            "status": "UNVERIFIED_NO_USABLE_AUTHOR_CENTER"})
        zero, aligned = float(row["zero_nll"]), float(row["aligned_nll"])
        improvement = zero - aligned
        assert improvement == float(row["improvement"])
        relations.append({**common, "zero_nll": zero, "aligned_nll": aligned,
            "improvement": improvement, "evidence_split": "frozen development experiment",
            "official_offset_relation_status": "UNVERIFIED_NO_USABLE_AUTHOR_CENTER",
            "center_error_relation_status": "UNVERIFIED_NO_USABLE_AUTHOR_CENTER"})
    assert len(rows) == 22 and sum(len(r["recording_ids"].split("|")) for r in rows) == 37
    write_csv("ln_vs_official_centers.csv", rows)
    write_csv("alignment_prediction_relation.csv", relations)
    offsets = [row["ln_radial_degree"] for row in relations]
    improvements = [row["improvement"] for row in relations]
    pearson = statistics.correlation(offsets, improvements)
    spearman = statistics.correlation(ranks(offsets), ranks(improvements))
    assert abs(pearson - old_summary["pearson_offset_improvement"]) < 1e-12
    assert abs(spearman - old_summary["spearman_offset_improvement"]) < 1e-12
    group_counts = {}
    for group in ["MC_ON", "MC_OFF", "PC_ON", "PC_OFF", "MC", "PC"]:
        count = sum(group in (row["retinal_class"], row["retinal_class"] + "_" + row["polarity"])
                    for row in rows)
        group_counts[group] = {"frozen_n": count, "usable_official_n": 0,
                               "comparison_status": "UNVERIFIED"}
    dump("comparison_summary.json", {
        "outcome": "FUNCTIONAL ALIGNMENT SUPPORTED, BIOLOGICAL CENTER UNRESOLVED",
        "usable_author_cell_count": 0, "frozen_cell_count": 22,
        "direction_and_magnitude_comparisons": "UNVERIFIED_NO_USABLE_AUTHOR_CENTER",
        "official_offset_vs_improvement": {"n": 0, "pearson": None, "spearman": None},
        "center_error_vs_improvement": {"n": 0, "pearson": None, "spearman": None},
        "LN_offset_vs_improvement": {"n": 22, "pearson": pearson, "spearman": spearman,
                                    "method": "Recomputed from frozen paired table only; no training or new prediction."},
        "mean_improvement": statistics.mean(improvements),
        "improved_cells": sum(value > 0 for value in improvements),
        "groups": group_counts,
    })
    dump("coordinate_verification.json", {
        "LN": {"raw_unit": "pooled pixel", "raw_x_positive": "image column right",
            "raw_y_positive": "image row down", "common_x_positive": "right",
            "common_y_positive": "up", "degree_per_pooled_pixel": 3 * 4.6 / 256,
            "transform": "x_degree=x_LN*3*4.6/256; y_degree=-y_LN*3*4.6/256",
            "sources": ["baselines/center_surround_ln.py:39-41", "data/schottdorf_lee_2021.py:220-229",
                        "output/audits/macaque_fixed_alignment_experiment_20260905/coordinate_mapping.json"],
            "all_22_checkpoint_to_frozen_table_centers_match": True},
        "official": {"usable_target_centers": 0, "orientation_status": "UNVERIFIED",
            "transform_applied": False, "coordinate_sign_selected_using_LN": False,
            "example_raw_fields": ["MxFr", "MxX", "MxY", "Max"],
            "example_index_origin": "UNVERIFIED", "example_y_axis": "UNVERIFIED",
            "native_nominal_degree_per_pixel": 4.6 / 256,
            "nominal_scale_is_not_a_verified_example_coordinate_transform": True,
            "author_prediction_model_grid": "linspace(-128,128,256), step 256/255; not evidence for RC-map index origin",
            "near_zero_vector_angles": "None computed; no eligible pairs. No threshold selected from LN agreement."},
        "frame_750_751": {"status": "UNVERIFIED_NOT_RECOMPUTED",
            "current_contract": "_LIVE_START_FRAME=751; acquisition zero remains unresolved in CURRENT_STATE.md:22",
            "author_readme": "751 blank frames, live starts at frame 752 in one-based notation",
            "author_notebook_timing_is_prediction_fitting_not_RC_sync": True,
            "conditional_analysis": "A uniform temporal lag translation can preserve a spatial peak only with unchanged boundaries, lag coverage and peak selection. These conditions are not verified for the author estimator.",
            "material_spatial_sensitivity_ruled_out": False,
            "two_alignment_displacement_degree": None,
            "why_no_two_alignment_run": "Author RF-estimation reproduction gate not met; running an invented estimator twice would not answer sensitivity of the author method.",
            "production_frame_zero_modified": False},
        "new_training_runs": 0, "RF_estimation_reproduction_runs": 0,
    })
    dump("reproduction_gate.json", {
        "decision": "NO_GO_NOT_AUTHORIZED_BY_CONDITIONAL_REPRODUCTION_CRITERIA",
        "raw_current_lineage_movie_spikes_available": True,
        "separate_author_6x1_video_payload_in_public_snapshot": False,
        "author_method_sufficiently_fixed": False,
        "could_avoid_validation_NLL_selection": True,
        "no_arbitrary_smoothing_peak_or_coordinate_choice_possible": False,
        "reported_example_reproduced_by_fixed_author_method": False,
        "missing": ["Executable reverse-correlation/deconvolution pipeline",
            "Exact MPEG spectral-peak truncation and temporal spectrum/window construction",
            "Deterministic polarity-aware temporal-peak selection and smoothing rule",
            "2D DoG fitting implementation and initialization/bounds/fit region",
            "Verified saved-map coordinate origin and orientation",
            "Author recording-selection or repeated-recording center-combination rule"],
        "PROTOCOL_created": False, "RF_center_recomputations": 0,
    })
    source_rows = list(csv.DictReader((OUT / "source_inventory.csv").open(encoding="utf-8-sig")))
    external = [
        ("paper.html", "https://pmc.ncbi.nlm.nih.gov/articles/PMC8998785/", "READ", "Original full paper; Methods, RC fitting, centering statistics and model optimization."),
        ("paper_bioc.xml", "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/PMC8998785/unicode", "READ", "Official NCBI text corroboration."),
        ("official_repository.html", "https://gin.g-node.org/Manuel/Macaque-ganglion-cells", "READ", "Live repository HEAD matches local snapshot."),
        ("official_archive.html", "https://gin.g-node.org/doi/Macaque-ganglion-cells", "READ", "Published archive root inventory; not a full archive payload scan."),
        ("Statistical_Summary_Document.xlsx.response.html", "https://pmc.ncbi.nlm.nih.gov/articles/instance/8998785/bin/NIHMS1782518-supplement-Statistical_Summary_Document.xlsx", "UNVERIFIED_ACCESS_BLOCKED", "HTML download challenge, not XLSX. Browser initially displayed reCAPTCHA, then blank; no readable workbook obtained."),
        ("", "https://physoc.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1113%2FJP281200&file=tjp14666-sup-0006-s08.xlsx", "UNVERIFIED_HTTP_403", "Publisher identifies 55.2 KB Statistical Summary Document; payload not inspected."),
        ("statistical_epmc.zip.response.html", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8998785/supplementaryFiles", "UNAVAILABLE_VIA_THIS_API", "API error says article not open access; not a supplementary ZIP."),
    ]
    for filename, url, status, note in external:
        path = OUT / filename if filename else None
        row = dict.fromkeys(source_rows[0], "")
        row.update(source_artifact=filename, official_url=url, access_status=status,
                   note=note, searched_fields="Required centering/localization fields" if status == "READ" else "NOT_SEARCHED_PAYLOAD_UNAVAILABLE")
        if path is not None:
            row.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       payload_path=str(path.relative_to(ROOT)))
        source_rows.append(row)
    write_csv("source_inventory.csv", source_rows)
    print(f"22 rows preserved; official matches=0; frozen LN-offset correlations={pearson:.12f}, {spearman:.12f}")


if __name__ == "__main__":
    main()
