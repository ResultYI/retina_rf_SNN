# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.2.6"]
# ///
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SEED = 2026090501


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def average_ranks(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(pearson=None, spearman=None)
    return dict(pearson=float(np.corrcoef(x, y)[0, 1]),
                spearman=float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1]))


def main() -> None:
    assert json.loads((OUT / "evaluation_status.json").read_text())["status"] == "COMPLETE"
    assert (OUT / "TEST_CONSUMED.md").exists()
    assert not (OUT / "analysis_summary.json").exists()
    protocol = json.loads((OUT / "protocol_hash.json").read_text())["sha256"]
    assert hashlib.sha256((OUT / "TEST_PROTOCOL.md").read_bytes()).hexdigest() == protocol
    rows = read_csv(OUT / "per_cell_test_nll.csv")
    pathways = read_csv(OUT / "pathway_per_cell.csv")
    assert len(rows) == 22 and len(pathways) == 66
    assert all(r["protocol_sha256"] == protocol for r in rows + pathways)
    cell_order = [r["cell_id"] for r in rows]
    initial = json.loads((OUT / "evidence_manifest.initial.json").read_text())
    zero = ROOT / initial["lineages"]["zero_center_reference"]
    assert cell_order == [c["cell_id"] for c in json.loads((zero / "results.json").read_text())["cells"]]
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, 22, size=(100000, 22))

    def paired(values: np.ndarray) -> dict:
        assert values.shape == (22,) and np.isfinite(values).all()
        draw = values[indices]
        mean_ci = np.percentile(draw.mean(axis=1), [2.5, 97.5], method="linear")
        median_ci = np.percentile(np.median(draw, axis=1), [2.5, 97.5], method="linear")
        return dict(cells=22, mean_delta_nll=float(values.mean()), median_delta_nll=float(np.median(values)),
            mean_ci_lower=float(mean_ci[0]), mean_ci_upper=float(mean_ci[1]),
            median_ci_lower=float(median_ci[0]), median_ci_upper=float(median_ci[1]),
            positive_count=int((values > 0).sum()), negative_count=int((values < 0).sum()), zero_count=int((values == 0).sum()),
            bootstrap_seed=SEED, bootstrap_draws=100000, protocol_sha256=protocol)

    comparisons = []
    for name, field in (("aligned-minus-zero", "aligned_minus_zero"), ("aligned-minus-LN", "aligned_minus_ln"), ("aligned-minus-CNN", "aligned_minus_cnn")):
        summary = paired(np.array([float(r[field]) for r in rows]))
        comparisons.append(dict(comparison=name, **summary, aligned_wins=summary["negative_count"],
            comparator_wins=summary["positive_count"], exact_ties=summary["zero_count"]))
    write_csv("paired_model_comparisons.csv", comparisons)
    audit = ROOT / initial["lineages"]["fixed_alignment_audit"]
    offset_rows = {r["cell_id"]: r for r in read_csv(audit / "per_cell_results.csv")}
    offsets = np.array([float(offset_rows[cid]["radial_offset_degrees"]) for cid in cell_order])
    improvement = -np.array([float(r["aligned_minus_zero"]) for r in rows])
    drop = max(sorted(range(22), key=lambda i: cell_order[i]), key=lambda i: abs(improvement[i]))
    keep = np.arange(22) != drop
    relation_rows = [dict(cell_id=r["cell_id"], group=r["group"], radial_offset_degrees=float(offsets[i]),
        zero_nll=float(r["zero_nll"]), aligned_nll=float(r["aligned_nll"]), improvement=float(improvement[i]),
        excluded_in_single_outlier_diagnostic=i == drop, offset_source=(audit / "per_cell_results.csv").relative_to(ROOT).as_posix(),
        protocol_sha256=protocol) for i, r in enumerate(rows)]
    write_csv("alignment_heldout_relation.csv", relation_rows)
    relation = dict(all_cells=dict(cells=22, mean_improvement=float(improvement.mean()), **correlation(offsets, improvement)),
        single_outlier_diagnostic=dict(excluded_cell=cell_order[drop], excluded_improvement=float(improvement[drop]),
            cells=21, mean_improvement=float(improvement[keep].mean()), **correlation(offsets[keep], improvement[keep])))
    development = json.loads((audit / "summary.json").read_text())
    write_csv("development_vs_heldout.csv", [
        dict(split="development", mean_aligned_minus_zero=development["mean_delta_nll"], aligned_wins=development["aligned_wins"],
            pearson_offset_improvement=development["pearson_offset_improvement"], spearman_offset_improvement=development["spearman_offset_improvement"]),
        dict(split="held-out", mean_aligned_minus_zero=comparisons[0]["mean_delta_nll"], aligned_wins=comparisons[0]["aligned_wins"],
            pearson_offset_improvement=relation["all_cells"]["pearson"], spearman_offset_improvement=relation["all_cells"]["spearman"]),
    ])
    populations, logit_rows, logit_correlations = [], [], []
    for pathway in ("H1", "direct-BC", "AC"):
        selected = {r["cell_id"]: r for r in pathways if r["pathway"] == pathway}
        ordered = [selected[cid] for cid in cell_order]
        delta = np.array([float(r["delta_nll"]) for r in ordered])
        summary = paired(delta)
        verdict = "HELD-OUT PREDICTIVE CONSEQUENCE SUPPORTED" if summary["mean_ci_lower"] > 0 else (
            "NO POSITIVE PREDICTIVE SUPPORT" if summary["mean_delta_nll"] <= 0 else "WEAK / MIXED")
        group_means = {group: float(np.mean([float(r["delta_nll"]) for r in ordered if r["group"] == group]))
                       for group in ("MC_ON", "MC_OFF", "PC_ON", "PC_OFF")}
        populations.append(dict(pathway=pathway, **summary, verdict=verdict, **{g+"_descriptive_mean": v for g, v in group_means.items()}))
        absolute = np.array([float(r["mean_abs_delta_logit"]) for r in ordered])
        logit_correlations.append(dict(pathway=pathway, cells=22, x="mean_abs_delta_logit", y="delta_nll",
            **correlation(absolute, delta), descriptive_only=True, protocol_sha256=protocol))
        logit_rows += [dict(cell_id=r["cell_id"], group=r["group"], pathway=pathway,
            mean_abs_delta_logit=float(r["mean_abs_delta_logit"]), signed_delta_logit=float(r["signed_mean_delta_logit"]),
            delta_nll=float(r["delta_nll"]), protocol_sha256=protocol) for r in ordered]
    write_csv("pathway_population.csv", populations)
    write_csv("pathway_logit_vs_nll.csv", logit_rows)
    write_csv("pathway_logit_nll_correlations.csv", logit_correlations)
    summary = dict(protocol_sha256=protocol, equal_cell_model_means={field: float(np.mean([float(r[field]) for r in rows]))
        for field in ("zero_nll", "aligned_nll", "ln_nll", "cnn_nll", "constant_nll")},
        paired_comparisons=comparisons, offset_relation=relation, pathways=populations,
        logit_nll_correlations=logit_correlations, bootstrap_index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        development_source=(audit / "summary.json").relative_to(ROOT).as_posix(), constant_is_test_fitted_descriptive=True,
        new_training_runs=0, new_model_seeds=0, p_values_computed=0)
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
