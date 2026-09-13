# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy", "pydantic", "opencv-python"]
# ///
# How to run: D:/anaconda/python.exe -B -u summarize_evidence.py
from __future__ import annotations

import csv
from itertools import combinations
import json
import math
from pathlib import Path
import statistics

import torch

from inference_diagnostics import OUT, PRIMARY, ROOT, SEEDS, SELECTED, SYN, ZERO

type Scalar = str | float | int | bool | None
type Row = dict[str, Scalar]
type Tensors = dict[str, torch.Tensor]


def write_rows(filename: str, rows: list[Row]) -> None:
    with (OUT / filename).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def similarity(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    assert a.shape == b.shape
    na, nb = float(a.norm()), float(b.norm())
    return {"cosine": float(a @ b) / (na * nb) if na * nb else math.nan,
            "relative_l2": float((b - a).norm()) / na if na else math.nan,
            "norm_A": na, "norm_B": nb, "rmse": float((b - a).square().mean().sqrt())}


def ranks(values: list[float]) -> list[float]:
    return [sum(x < value for x in values) + (sum(x == value for x in values) + 1) / 2 for value in values]


def rank_correlation(a: list[float], b: list[float]) -> float | None:
    x, y = torch.tensor(ranks(a), dtype=torch.float64), torch.tensor(ranks(b), dtype=torch.float64)
    x, y = x - x.mean(), y - y.mean()
    return similarity(x, y)["cosine"] if x.norm() * y.norm() else None


def moments(v: torch.Tensor) -> dict[str, float]:
    x = v.detach().double().flatten()
    return {"mean_abs": float(x.abs().mean()), "signed_mean": float(x.mean()),
            "rms": float(x.square().mean().sqrt()), "positive_fraction": float((x > 0).double().mean())}


def rf_shape(v: torch.Tensor, positions: torch.Tensor) -> dict[str, Scalar]:
    x, p = v.detach().double(), positions.double()
    energy = x.square().sum(tuple(range(x.ndim - 1)))
    center = (energy[:, None] * p).sum(0) / energy.sum()
    extent = ((p - center).square().sum(-1) * energy).sum().div(energy.sum()).sqrt()
    temporal = x.sum(-1).mean(tuple(range(x.ndim - 2))) if x.ndim > 2 else x.sum(-1)
    temporal_energy = x.square().sum(-1)
    if temporal_energy.ndim > 1:
        temporal_energy = temporal_energy.mean(tuple(range(temporal_energy.ndim - 1)))
    return {"center_x_deg": float(center[0]), "center_y_deg": float(center[1]), "rms_extent_deg": float(extent),
            "temporal_signed_marginal": json.dumps(temporal.tolist()),
            "temporal_energy_marginal": json.dumps(temporal_energy.tolist())}


def main() -> None:
    torch.set_num_threads(2)
    assert not (OUT / "rf_identifiability.csv").exists()
    diag = torch.load(OUT / "inference-results.pt", weights_only=True)
    validation = torch.load(OUT / "validation-replay.pt", weights_only=True)
    rfs = torch.load(PRIMARY / "rf-tensors.pt", weights_only=True)
    effects = torch.load(PRIMARY / "perturbation-tensors.pt", weights_only=True)
    params = torch.load(PRIMARY / "effective-parameters.pt", weights_only=True)
    syn_rf = torch.load(SYN / "rf-tensors.pt", weights_only=True)
    syn_cf = torch.load(SYN / "counterfactual-tensors.pt", weights_only=True)
    syn_geometry = torch.load(SYN / "teacher.pt", weights_only=True)["cone_positions"]
    rf_rows, path_rows, pairs, temporal_rows, retention_rows = [], [], [], [], []

    def compare(cohort: str, cid: str, label_a: str, label_b: str, logits_a: torch.Tensor,
                logits_b: torch.Tensor, nll_a: float, nll_b: float, rf_a: Tensors, rf_b: Tensors,
                delta_a: Tensors, delta_b: Tensors, mask: torch.Tensor, positions: torch.Tensor,
                parameter_a: Tensors | None = None, parameter_b: Tensors | None = None) -> None:
        identity: Row = {"cohort": cohort, "cell_id": cid, "fit_A": label_a, "fit_B": label_b}
        prediction = similarity(logits_a[mask], logits_b[mask])
        pair = identity | {"NLL_A": nll_a, "NLL_B": nll_b, "delta_validation_NLL": nll_b - nll_a,
            "prediction_logit_cosine": prediction["cosine"], "prediction_logit_RMSE": prediction["rmse"],
            "global_RF_cosine": similarity(rf_a["global"], rf_b["global"])["cosine"]}
        for pathway in ("global", "H1", "direct_BC", "AC"):
            a, b = rf_a[pathway], rf_b[pathway]
            shape_a, shape_b = rf_shape(a, positions), rf_shape(b, positions)
            center_a = torch.tensor([shape_a["center_x_deg"], shape_a["center_y_deg"]], dtype=torch.float64)
            center_b = torch.tensor([shape_b["center_x_deg"], shape_b["center_y_deg"]], dtype=torch.float64)
            nonzero = (a != 0) & (b != 0)
            rf_rows.append(identity | {"pathway": pathway, **similarity(a, b),
                "center_displacement_deg": float((center_b - center_a).norm()),
                "temporal_marginal_cosine": similarity(a.sum(-1), b.sum(-1))["cosine"],
                "nonzero_element_sign_agreement": float(((a.sign() == b.sign())[nonzero]).double().mean()),
                **{f"{key}_A": value for key, value in shape_a.items()},
                **{f"{key}_B": value for key, value in shape_b.items()}})
        for pathway in ("H1", "direct_BC", "AC"):
            a, b = delta_a[pathway][mask], delta_b[pathway][mask]
            m_a, m_b, sim = moments(a), moments(b), similarity(a, b)
            ratio = m_b["mean_abs"] / m_a["mean_abs"]
            nonzero = (a != 0) & (b != 0)
            path_rows.append(identity | {"pathway": pathway, **sim, "magnitude_ratio_B_over_A": ratio,
                "signed_mean_same_sign": m_a["signed_mean"] * m_b["signed_mean"] > 0,
                "nonzero_bin_sign_agreement": float(((a.sign() == b.sign())[nonzero]).double().mean()),
                **{f"{key}_A": value for key, value in m_a.items()},
                **{f"{key}_B": value for key, value in m_b.items()}})
            pair.update({f"{pathway}_intervention_cosine": sim["cosine"], f"{pathway}_magnitude_ratio": ratio})
        if parameter_a is not None and parameter_b is not None:
            differences = [(parameter_b[k].double() - parameter_a[k].double()).flatten()
                           / (bounds[..., 1] - bounds[..., 0]).double().flatten()
                           for k, bounds in diag["bounds"][cid].items()]
            pair["tau_delay_distance_RMS_bound_width"] = float(torch.cat(differences).square().mean().sqrt())
            pair["temporal_distance_status"] = "effective values; BC labels retained"
        else:
            pair["temporal_distance_status"] = "CURRENT-CONTRACT TEMPORAL PARAMETER RECOVERY NOT TESTED"
        pairs.append(pair)

    zero_results = {r["cell_id"]: r for r in json.loads((ZERO / "results.json").read_text())["cells"]}
    aligned_results = {r["cell_id"]: r for r in json.loads((PRIMARY / "results.json").read_text())["cells"]}
    for cid, data in validation.items():
        zero = torch.load(ZERO / "cells" / cid.replace("#", "_") / "validation-predictions.pt", weights_only=True)
        assert torch.equal(zero["target"], data["events"]) and torch.equal(zero["valid_mask"], data["mask"])
        assert tuple(zero["source_image_ids"]) == tuple(data["source_image_ids"])
        assert tuple(zero["trial_indices"]) == tuple(data["trial_indices"])
        compare("alignment_nuisance", cid, "zero", "aligned", zero["logits_trained"], data["logits"],
            zero_results[cid]["validation_nll_trained"], aligned_results[cid]["validation_nll_trained"],
            rfs[cid]["zero"], rfs[cid]["aligned"],
            {p: effects[cid]["zero"][p + "_off"]["logit_delta"] for p in ("H1", "direct_BC", "AC")},
            {p: effects[cid]["aligned"][p + "_off"]["logit_delta"] for p in ("H1", "direct_BC", "AC")},
            data["mask"], data["cone_positions"], params[cid]["zero"], params[cid]["aligned"])
    for cid in SELECTED:
        fits = {}
        for label in ("primary", "fresh_1", "fresh_2"):
            folder = SEEDS / "fits" / cid.replace("#", "_") / label
            fits[label] = (torch.load(folder / "evaluation.pt", weights_only=True)["validation_logits"],
                           json.loads((folder / "results.json").read_text())["validation_nll"])
        for a, b in combinations(fits, 2):
            values_a, nll_a = fits[a]
            values_b, nll_b = fits[b]
            compare("zero_center_seed", cid, a, b, values_a["normal"], values_b["normal"], nll_a, nll_b,
                diag["seed_rf"][cid][a], diag["seed_rf"][cid][b],
                {p: values_a[p + "_off"] - values_a["normal"] for p in ("H1", "direct_BC", "AC")},
                {p: values_b[p + "_off"] - values_b["normal"] for p in ("H1", "direct_BC", "AC")},
                validation[cid]["mask"], validation[cid]["cone_positions"],
                diag["parameters"][cid][a], diag["parameters"][cid][b])
        for pathway in ("H1", "direct_BC", "AC"):
            magnitudes = [moments((v[pathway + "_off"] - v["normal"])[validation[cid]["mask"]])["mean_abs"]
                          for v, _ in fits.values()]
            path_rows.append({"cohort": "zero_center_seed_summary", "cell_id": cid, "pathway": pathway,
                "magnitude_min": min(magnitudes), "magnitude_max": max(magnitudes),
                "magnitude_range": max(magnitudes) - min(magnitudes),
                "magnitude_sample_SD": statistics.stdev(magnitudes),
                "magnitude_CV": statistics.stdev(magnitudes) / statistics.mean(magnitudes)})
    for seed in ("53001", "53002", "53003"):
        compare("historical_shared_BC_synthetic", "population_8", "teacher", seed,
            diag["synthetic_logits"]["teacher"], diag["synthetic_logits"][seed],
            diag["synthetic_metrics"]["teacher"]["expected_ce"], diag["synthetic_metrics"][seed]["expected_ce"],
            syn_rf["teacher"], syn_rf[seed + "_trained"],
            {p: syn_cf["teacher"][p + "_off"]["logit_delta"] for p in ("H1", "direct_BC", "AC")},
            {p: syn_cf[seed][p + "_off"]["logit_delta"] for p in ("H1", "direct_BC", "AC")},
            torch.ones_like(diag["synthetic_logits"]["teacher"], dtype=torch.bool), syn_geometry)
        for p in ("global", "H1", "direct_BC", "AC"):
            rf_rows.append({"cohort": "historical_shared_BC_synthetic_raw", "cell_id": "population_8", "fit_A": "teacher",
                "fit_B": seed + "_raw", "pathway": p, **similarity(syn_rf["teacher"][p], syn_rf[seed + "_raw"][p])})
        for p in ("H1", "direct_BC", "AC"):
            path_rows.append({"cohort": "historical_shared_BC_synthetic_RF_intervention", "cell_id": "population_8", "fit_A": "teacher",
                "fit_B": seed, "pathway": p,
                **similarity(syn_cf["teacher"][p + "_off"]["rf_delta"], syn_cf[seed][p + "_off"]["rf_delta"])})

    for window, source in ((150, diag["aligned_full150"]), (600, diag["aligned_full600"])):
        for cid, bundle in source.items():
            for pathway, values in (bundle if window == 150 else bundle["rf"]).items():
                energy = values.double().square().sum((1, 3))
                context_total = energy.sum(-1)
                fractions = energy[:, -16:].sum(-1) / context_total
                retention_rows.append({"cell_id": cid, "pathway": pathway, "window_bins": window,
                    "contexts": values.shape[0], "retained_energy_pooled": float(energy[:, -16:].sum() / energy.sum()),
                    "retained_energy_min_context": float(fractions.min()), "retained_energy_max_context": float(fractions.max()),
                    "retained_norm_fraction_pooled": float((energy[:, -16:].sum() / energy.sum()).sqrt()),
                    "oldest_100_energy_fraction": float(energy[:, :100].sum() / energy.sum()) if window == 600 else None})

    def parameter_rows(cid: str, fit: str, values: Tensors, cohort: str) -> None:
        for key, value in values.items():
            bounds = diag["bounds"][cid].get(key)
            for index, theta in enumerate(value.flatten().tolist()):
                row: Row = {"cohort": cohort, "cell_id": cid, "fit": fit, "quantity": key, "mode": index, "effective_value": theta}
                if bounds is not None:
                    low, high = bounds.reshape(-1, 2)[index].tolist()
                    active_low = low
                    partner = key.replace("BC_sustained", "BC_transient").replace("AC_local", "AC_transient")
                    if partner != key:
                        active_low = max(low, float(values[partner].flatten()[index]) + torch.finfo(torch.float32).eps * high)
                    u = (theta - low) / (high - low)
                    row.update({"lower_bound": low, "upper_bound": high, "u": u, "active_lower_bound": active_low,
                        "active_u": (theta - active_low) / (high - active_low),
                        "boundary": "lower" if u <= .05 else "upper" if u >= .95 else "interior", "unit": "ms"})
                temporal_rows.append(row)
    for cid in validation:
        for fit in ("zero", "aligned"):
            parameter_rows(cid, fit, params[cid][fit], "alignment_value")
    for cid in SELECTED:
        for fit, values in diag["parameters"][cid].items():
            parameter_rows(cid, fit, values, "zero_center_seed_value")
    for key, first in next(iter(params.values()))["aligned"].items():
        for index in range(first.numel()):
            a = [float(v["zero"][key].flatten()[index]) for v in params.values()]
            b = [float(v["aligned"][key].flatten()[index]) for v in params.values()]
            delta = [y - x for x, y in zip(a, b, strict=True)]
            temporal_rows.append({"cohort": "alignment_summary", "quantity": key, "mode": index,
                "median_zero": statistics.median(a), "median_aligned": statistics.median(b),
                "median_abs_change": statistics.median(map(abs, delta)),
                "median_relative_change": statistics.median(abs(d) / abs(x) for x, d in zip(a, delta, strict=True) if x != 0),
                "increase_count": sum(d > 0 for d in delta), "decrease_count": sum(d < 0 for d in delta),
                "rank_correlation": rank_correlation(a, b)})
    for cid in SELECTED:
        nlls = [json.loads((SEEDS / "fits" / cid.replace("#", "_") / fit / "results.json").read_text())["validation_nll"]
                for fit in ("primary", "fresh_1", "fresh_2")]
        for key, first in diag["parameters"][cid]["primary"].items():
            for index in range(first.numel()):
                values = [float(v[key].flatten()[index]) for v in diag["parameters"][cid].values()]
                mean = statistics.mean(values)
                temporal_rows.append({"cohort": "zero_center_seed_summary", "cell_id": cid, "quantity": key, "mode": index,
                    "minimum": min(values), "maximum": max(values), "range": max(values) - min(values),
                    "sample_SD": statistics.stdev(values), "CV": statistics.stdev(values) / abs(mean) if mean else None,
                    "NLL_spread": max(nlls) - min(nlls)})
    basis_rows = []
    for cid, values in diag["bc_temporal_basis"].items():
        for family, v in zip(("sustained", "transient"), values, strict=True):
            for a, b in combinations(range(3), 2):
                basis_rows.append({"cell_id": cid, "family": family, "mode_A": a, "mode_B": b,
                                   **similarity(v[a], v[b])})
    write_rows("rf_identifiability.csv", rf_rows)
    write_rows("pathway_identifiability.csv", path_rows)
    write_rows("temporal_identifiability.csv", temporal_rows)
    write_rows("prediction_vs_mechanism_equivalence.csv", pairs)
    write_rows("rf_truncation.csv", retention_rows)
    write_rows("bc_temporal_basis_overlap.csv", basis_rows)
    order_rows = []
    for cid, fits in diag["parameters"].items():
        for key in ("tau_BC_sustained_basis", "tau_BC_transient_basis"):
            orders = [tuple(torch.argsort(values[key]).tolist()) for values in fits.values()]
            order_rows.append({"cell_id": cid, "quantity": key, "primary_order": str(orders[0]),
                "fresh_1_order": str(orders[1]), "fresh_2_order": str(orders[2]),
                "rank_order_stable": len(set(orders)) == 1, "constraint": "within-family order not enforced"})
    write_rows("temporal_order_stability.csv", order_rows)
    print(json.dumps({"RF_rows": len(rf_rows), "pathway_rows": len(path_rows), "temporal_rows": len(temporal_rows),
        "pair_rows": len(pairs), "retention_rows": len(retention_rows)}, indent=2))


if __name__ == "__main__":
    main()
