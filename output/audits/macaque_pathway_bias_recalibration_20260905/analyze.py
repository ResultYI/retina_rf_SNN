# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy==2.2.6"]
# ///
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from control import OUT, OLD, csv_out, json_out, nll, protocol_hash, read_csv, sha


def main() -> None:
    digest = protocol_hash()
    lock = json.loads((OUT / "bias_fit_lock.json").read_text())
    assert lock["protocol_sha256"] == digest
    assert sha(OUT / "bias_fit_per_cell.csv") == lock["bias_fit_per_cell_sha256"]
    assert not (OUT / "analysis_summary.json").exists()
    torch.set_num_threads(2)
    fits = read_csv(OUT / "bias_fit_per_cell.csv")
    lookup = {(r["cell_id"], r["condition"]): r for r in fits}
    order = list(dict.fromkeys(r["cell_id"] for r in fits))
    assert len(order) == 22 and len(fits) == 88
    normal_rows, pathway_rows, components = [], [], []
    raw_precision_errors = []
    for cid in order:
        saved = torch.load(OLD / "cells" / (cid.replace("#", "_") + "-test-predictions.pt"), weights_only=True)
        mask = saved["valid_mask"]
        target = saved["target"][mask].double()
        normal = saved["logits"]["aligned"][mask].double()
        normal_nll = nll(normal, target)
        normal_fit = lookup[cid, "normal"]
        normal_bias = float(normal_fit["fitted_bias"])
        normal_recal = nll(normal+normal_bias, target)
        normal_rows.append(dict(cell_id=cid, group=normal_fit["group"], fitted_bias=normal_bias,
            train_nll_before=float(normal_fit["train_nll_before"]), train_nll_after=float(normal_fit["train_nll_after"]),
            heldout_nll_before=normal_nll, heldout_nll_after=normal_recal, heldout_delta_nll=normal_recal-normal_nll,
            train_observed_rate=float(normal_fit["train_observed_rate"]),
            train_predicted_rate_after=float(normal_fit["train_predicted_rate_after"]),
            convergence_residual_mean_rate=float(normal_fit["convergence_residual_mean_rate"]), diagnostic_only=True))
        for key, tensor in saved["logits"].items():
            if key not in {"aligned", "H1-off", "direct-BC-off", "AC-off"}:
                continue
            original32 = float(((torch.nn.functional.softplus(tensor)-saved["target"]*tensor)*mask).sum()/mask.sum())
            raw_precision_errors.append(abs(nll(tensor[mask].double(), target)-original32))
        for pathway in ("H1", "direct-BC", "AC"):
            fitted = lookup[cid, pathway]
            bias = float(fitted["fitted_bias"])
            off = saved["logits"][pathway+"-off"][mask].double()
            raw_nll, recal_nll = nll(off, target), nll(off+bias, target)
            raw_delta, recal_delta = raw_nll-normal_nll, recal_nll-normal_nll
            difference = off-normal
            signed, absolute = float(difference.mean()), float(difference.abs().mean())
            centered = difference-difference.mean()
            pathway_rows.append(dict(cell_id=cid, group=fitted["group"], pathway=pathway, fitted_bias=bias,
                train_observed_rate=float(fitted["train_observed_rate"]),
                train_predicted_rate_before=float(fitted["train_predicted_rate_before"]),
                train_predicted_rate_after=float(fitted["train_predicted_rate_after"]), heldout_normal_nll=normal_nll,
                heldout_off_nll_raw=raw_nll, heldout_off_nll_recal=recal_nll,
                raw_delta_nll=raw_delta, recal_delta_nll=recal_delta,
                fraction_raw_penalty_removed=1-recal_delta/raw_delta if raw_delta > 0 else None,
                signed_mean_delta_logit=signed, mean_abs_delta_logit=absolute, protocol_sha256=digest))
            components.append(dict(cell_id=cid, group=fitted["group"], pathway=pathway,
                centered_delta_logit_std=float(centered.std(correction=0)),
                centered_delta_logit_rms=float(centered.square().mean().sqrt()),
                raw_mean_delta_logit=signed, mean_abs_delta_logit=absolute, scored_bins=int(mask.sum()),
                standard_deviation_ddof=0, protocol_sha256=digest))
    indices = np.random.default_rng(2026090502).integers(0, 22, size=(100000, 22))
    populations = []
    for pathway in ("H1", "direct-BC", "AC"):
        selected = [r for r in pathway_rows if r["pathway"] == pathway]
        assert [r["cell_id"] for r in selected] == order
        raw = np.array([r["raw_delta_nll"] for r in selected])
        recal = np.array([r["recal_delta_nll"] for r in selected])
        sample = recal[indices]
        mean_ci = np.percentile(sample.mean(axis=1), [2.5, 97.5], method="linear")
        median_ci = np.percentile(np.median(sample, axis=1), [2.5, 97.5], method="linear")
        relative = 1-float(recal.mean()/raw.mean()) if raw.mean() > 0 else None
        positive = int((recal > 0).sum())
        if recal.mean() <= 0:
            verdict = "NO POSITIVE RESIDUAL SUPPORT"
        elif pathway in {"direct-BC", "AC"} and relative is not None and relative > 0.5 and mean_ci[0] <= 0 <= mean_ci[1]:
            verdict = "OPERATING-POINT DOMINATED"
        elif mean_ci[0] > 0 and positive >= 17 and relative is not None and relative < 0.5:
            verdict = "STIMULUS-DEPENDENT PREDICTIVE CONTRIBUTION SUPPORTED"
        else:
            verdict = "MIXED"
        populations.append(dict(pathway=pathway, cells=22, raw_mean_delta_nll=float(raw.mean()),
            recal_mean_delta_nll=float(recal.mean()), raw_median_delta_nll=float(np.median(raw)),
            recal_median_delta_nll=float(np.median(recal)), absolute_reduction=float(raw.mean()-recal.mean()),
            relative_reduction=relative, recal_mean_ci_lower=float(mean_ci[0]), recal_mean_ci_upper=float(mean_ci[1]),
            recal_median_ci_lower=float(median_ci[0]), recal_median_ci_upper=float(median_ci[1]),
            recal_positive_count=positive, recal_negative_count=int((recal < 0).sum()), recal_zero_count=int((recal == 0).sum()),
            bootstrap_seed=2026090502, bootstrap_draws=100000, verdict=verdict, protocol_sha256=digest))
    csv_out("normal_bias_sanity.csv", normal_rows)
    csv_out("pathway_bias_recalibration_per_cell.csv", pathway_rows)
    csv_out("pathway_bias_recalibration_population.csv", populations)
    csv_out("stimulus_dependent_logit_components.csv", components)
    normal_delta = np.array([r["heldout_delta_nll"] for r in normal_rows])
    normal_b = np.array([r["fitted_bias"] for r in normal_rows])
    comp_summary = {}
    for pathway in ("H1", "direct-BC", "AC"):
        rows = [r for r in components if r["pathway"] == pathway]
        comp_summary[pathway] = dict(median_abs_mean_shift=float(np.median([abs(r["raw_mean_delta_logit"]) for r in rows])),
            median_centered_std=float(np.median([r["centered_delta_logit_std"] for r in rows])))
    result = dict(protocol_sha256=digest, pathways=populations,
        normal_sanity=dict(mean_bias=float(normal_b.mean()), median_bias=float(np.median(normal_b)),
            min_bias=float(normal_b.min()), max_bias=float(normal_b.max()),
            train_mean_before=float(np.mean([r["train_nll_before"] for r in normal_rows])),
            train_mean_after=float(np.mean([r["train_nll_after"] for r in normal_rows])),
            test_mean_before=float(np.mean([r["heldout_nll_before"] for r in normal_rows])),
            test_mean_after=float(np.mean([r["heldout_nll_after"] for r in normal_rows])),
            mean_test_delta=float(normal_delta.mean()), median_test_delta=float(np.median(normal_delta)),
            test_improved_cells=int((normal_delta < 0).sum()), test_worsened_cells=int((normal_delta > 0).sum()),
            test_tied_cells=int((normal_delta == 0).sum())),
        solver=dict(scalars=88, max_abs_mean_rate_residual=max(abs(float(r["convergence_residual_mean_rate"])) for r in fits),
            max_iterations=max(int(r["iterations"]) for r in fits), min_iterations=min(int(r["iterations"]) for r in fits),
            max_train_nll_change=max(float(r["train_nll_after"])-float(r["train_nll_before"]) for r in fits)),
        centered_logit_descriptive=comp_summary, raw_float64_vs_prior_float32_max_abs_nll=max(raw_precision_errors),
        bootstrap_index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        bias_fit_hash_unchanged=sha(OUT / "bias_fit_per_cell.csv") == lock["bias_fit_per_cell_sha256"],
        evidence_level="predefined follow-up analysis on an already consumed confirmatory set", model_training=0)
    json_out("analysis_summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
