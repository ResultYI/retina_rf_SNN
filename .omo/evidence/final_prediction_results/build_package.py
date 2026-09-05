#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.2.6", "pydantic==2.8.2"]
# ///
# How to run: D:/anaconda/python.exe -B .omo/evidence/final_prediction_results/build_package.py
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

OUT: Final = Path(__file__).resolve().parent
ROOT: Final = OUT.parents[2]
CNN: Final = ROOT / ".omo/evidence/compact_causal_cnn_baseline"
LN: Final = ROOT / "output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830"
CANONICAL: Final = ROOT / "output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830"
SC: Final = ROOT / ".omo/evidence/spatial_contrast_adapted"
GROUPS: Final = ("MC_ON", "MC_OFF", "PC_ON", "PC_OFF")
MODELS: Final = ("Constant", "LN", "CNN", "Canonical V1", "SC-adapted")
PAIRS: Final = (("CNN", "LN"), ("CNN", "Canonical V1"), ("Canonical V1", "LN"))
BOOTSTRAP_SEED: Final = 20260831
BOOTSTRAP_REPLICATES: Final = 100_000
FloatArray = NDArray[np.float64]
Scalar = str | int | float


class Cell(BaseModel):
    model_config = ConfigDict(frozen=True)
    cell_id: str
    group: Literal["MC_ON", "MC_OFF", "PC_ON", "PC_OFF"]
    constant_nll: FiniteFloat = Field(ge=0)
    ln_nll: FiniteFloat = Field(ge=0)
    cnn_nll: FiniteFloat = Field(ge=0)
    canonical_v1_nll: FiniteFloat = Field(ge=0)
    sc_adapted_nll: FiniteFloat = Field(ge=0)
    train_bins: int = Field(gt=0)
    validation_bins: int = Field(gt=0)

    def values(self) -> tuple[float, ...]:
        return self.constant_nll, self.ln_nll, self.cnn_nll, self.canonical_v1_nll, self.sc_adapted_nll


@dataclass(frozen=True, slots=True)
class PairedSummary:
    mean: float
    median: float
    mean_ci_low: float
    mean_ci_high: float
    median_ci_low: float
    median_ci_high: float
    first_wins: int
    second_wins: int
    ties: int


def paired_summary(difference: FloatArray, indices: NDArray[np.int16]) -> PairedSummary:
    resampled = difference[indices]
    mean_ci = np.quantile(resampled.mean(axis=1), (0.025, 0.975), method="linear")
    median_ci = np.quantile(np.median(resampled, axis=1), (0.025, 0.975), method="linear")
    return PairedSummary(float(difference.mean()), float(np.median(difference)),
                         float(mean_ci[0]), float(mean_ci[1]), float(median_ci[0]), float(median_ci[1]),
                         int((difference < 0).sum()), int((difference > 0).sum()), int((difference == 0).sum()))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(name: str, rows: Sequence[Mapping[str, Scalar]]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(headers: Sequence[str], rows: Sequence[Sequence[Scalar]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
                     + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def main() -> None:
    sources = [CNN / "per_cell.csv", CNN / "results.json", CNN / "preflight.json",
               CNN / "training_complete.json", LN / "results.json", CANONICAL / "results.json", SC / "results.json"]
    sources += sorted((CNN / "cells").glob("*/results.json"))
    source_hashes = {str(path): sha256(path) for path in sources}
    with (CNN / "per_cell.csv").open(newline="", encoding="utf-8") as stream:
        cells = [Cell.model_validate(row) for row in csv.DictReader(stream)]
    assert len(cells) == len({c.cell_id for c in cells}) == 22
    assert [sum(c.group == g for c in cells) for g in GROUPS] == [5, 4, 9, 4]
    references = [json.loads((root / "results.json").read_text(encoding="utf-8")) for root in (CNN, LN, CANONICAL, SC)]
    maps = [{r["cell_id"]: r for r in doc["cells"]} for doc in references]
    assert all(set(m) == {c.cell_id for c in cells} for m in maps)
    preflight = json.loads((CNN / "preflight.json").read_text(encoding="utf-8"))
    assert references[0]["all_22_target_mask_order_exact"]
    assert references[3]["status"] == "COMPLETED_CORRECTED" and references[3]["frozen_benchmark_accepted"]
    for cell in cells:
        cnn, ln, canonical, sc = [m[cell.cell_id] for m in maps]
        assert cell == Cell.model_validate(cnn)
        assert (cell.constant_nll, cell.ln_nll) == (ln["constant_nll"], ln["ln_nll"])
        assert cell.canonical_v1_nll == canonical["validation_nll_trained"] == sc["canonical_v1_nll"]
        assert cell.sc_adapted_nll == sc["sc_adapted_nll"] and cell.ln_nll == sc["center_surround_ln_nll"]
        assert cell.group == ln["group"] == sc["group"] == f"{canonical['retinal_class']}_{canonical['polarity']}"
        assert cell.validation_bins == ln["validation_valid_bins"] == canonical["validation_valid_bins"] == sc["validation_bins"]
        assert cell.train_bins == ln["train_valid_bins"] == canonical["train_valid_bins"] == sc["train_bins"]
        assert canonical["parameter_counts"] == dict(total=129, requires_grad=33, optimizer_listed=33)
        assert ln["parameter_count"] == 128
        fit = json.loads((CNN / "cells" / cell.cell_id.replace("#", "_") / "results.json").read_text())
        assert all(fit["parameter_counts"][k] == 2990 for k in ("total", "requires_grad", "optimizer_listed"))
    for path in sources:
        if str(path) in preflight["source_sha256"]:
            assert source_hashes[str(path)] == preflight["source_sha256"][str(path)]
    matrix = np.array([c.values() for c in cells], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, 22, size=(BOOTSTRAP_REPLICATES, 22), dtype=np.int16)
    paired = []
    for first, second in PAIRS:
        difference = matrix[:, MODELS.index(first)] - matrix[:, MODELS.index(second)]
        result = paired_summary(difference, indices)
        assert abs(result.mean - statistics.fmean(difference.tolist())) < 1e-15
        assert result.median == statistics.median(difference.tolist())
        assert result.first_wins + result.second_wins + result.ties == 22
        paired.append(dict(comparison=f"{first} - {second}", **asdict(result)))
    np.savez_compressed(OUT / "bootstrap_resamples.npz", cell_indices=indices)
    rows = [dict(c.model_dump(), **{f"delta_{a}_{b}": c.values()[MODELS.index(a)] - c.values()[MODELS.index(b)]
                                  for a, b in PAIRS}) for c in cells]
    write_csv("per_cell_nll.csv", rows)
    write_csv("paired_comparisons.csv", paired)
    descriptive = []
    for group in ("overall", *GROUPS):
        selected = matrix if group == "overall" else matrix[[c.group == group for c in cells]]
        for index, model in enumerate(MODELS):
            values = selected[:, index]
            descriptive.append(dict(group=group, model=model, cells=len(values), mean=float(values.mean()),
                                    median=float(np.median(values)), sample_sd=float(values.std(ddof=1)),
                                    q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)),
                                    minimum=float(values.min()), maximum=float(values.max())))
    write_csv("group_descriptive.csv", descriptive)
    counts = [("Constant", 1, 1, "one training-estimated probability; analytical fit, no optimizer"),
              ("LN", 128, 128, "stored scalar coordinates; includes normalized temporal filters and bias/history"),
              ("CNN", 2990, 2990, "requires-grad and optimizer-listed scalars, including convolution biases and LN output/history head"),
              ("Canonical V1", 33, 129, "requires-grad and optimizer-listed scalars; 129 registered total parameters, excluding buffers")]
    parameters = [dict(model=m, fitted_scalars_per_cell=n, total_stored_parameter_scalars=total,
                       fitted_scalars_all_22=22*n, counting_scope=scope) for m, n, total, scope in counts]
    write_csv("main_parameters.csv", parameters)
    write_csv("supplement_sc_parameters.csv", [dict(model="SC-adapted", fitted_output_scalars=4,
              inherited_frozen_center_coordinates=64, inherited_plus_fitted_coordinates=68,
              counting_scope="raw coordinate bookkeeping, not independent functional degrees of freedom; no history")])
    headline = [[m, f"{matrix[:, j].mean():.9f}", f"{np.median(matrix[:, j]):.9f}"] for j, m in enumerate(MODELS[:4])]
    pair_rows = [[r["comparison"], f"{r['mean']:.9f}", f"[{r['mean_ci_low']:.9f}, {r['mean_ci_high']:.9f}]",
                  f"{r['median']:.9f}", f"[{r['median_ci_low']:.9f}, {r['median_ci_high']:.9f}]",
                  f"{r['first_wins']} / {r['second_wins']} / {r['ties']}"] for r in paired]
    group_rows = [[g.replace("_", " "), sum(c.group == g for c in cells)] +
                  [f"{r['mean']:.9f}" for r in descriptive if r['group'] == g and r['model'] in MODELS[:4]] for g in GROUPS]
    main_tables = ["# Final prediction tables", "NLL unit: nats per valid Bernoulli bin; lower is better. Equal cell weight.",
                   "## Overall (22 cells)", markdown(["Model", "Mean NLL", "Median NLL"], headline),
                   "## Paired differences", "Difference = first minus second; wins = strictly lower NLL, exact ties only.",
                   markdown(["Comparison", "Mean", "Mean 95% CI", "Median", "Median 95% CI", "First / second / ties"], pair_rows),
                   "## Cell groups (descriptive only)", markdown(["Group", "Cells", *MODELS[:4]], group_rows),
                   "## Parameter counts", markdown(["Model", "Fitted scalars/cell", "Registered total/cell"],
                   [[m, n, total] for m, n, total, _ in counts]),
                   "33 and 2990 count trainable scalar coordinates, not independent functional DoF or actually-updated counts. "
                   "Frozen parameters/buffers are not trainable capacity. This is not a matched-capacity comparison.",
                   "## Per cell", markdown(["Cell", "Group", *MODELS[:4]],
                   [[c.cell_id, c.group.replace('_', ' '), *[f"{v:.9f}" for v in c.values()[:4]]] for c in cells])]
    (OUT / "tables.md").write_text("\n\n".join(main_tables) + "\n", encoding="utf-8")
    supplement = ["# Supplement: frozen SC-adapted", "No SC-adapted inference test or new fit was performed.",
                  "## Descriptive NLL", markdown(["Group", "Cells", *MODELS],
                  [[g.replace('_', ' '), 22 if g == 'overall' else sum(c.group == g for c in cells)] +
                   [f"{r['mean']:.9f}" for r in descriptive if r['group'] == g] for g in ("overall", *GROUPS)]),
                  "## Per cell", markdown(["Cell", "Group", *MODELS],
                  [[c.cell_id, c.group.replace('_', ' '), *[f"{v:.9f}" for v in c.values()]] for c in cells]),
                  "SC-adapted retains the frozen LN center filter: 64 inherited coordinates plus four fitted output parameters. "
                  "It has no spike-history term. The count 68 is raw bookkeeping, not independent DoF; it is not a faithful original SC fit."]
    (OUT / "supplement.md").write_text("\n\n".join(supplement) + "\n", encoding="utf-8")
    method = dict(replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED, generator="NumPy PCG64",
                  unit="cell", sample_size=22, replacement=True, paired=True, stratified=False,
                  interval="two-sided percentile 2.5/97.5; NumPy linear quantiles", estimands=["mean difference", "median difference"],
                  shared_indices_across_comparisons=True, multiple_comparison_adjustment=False,
                  group_significance_tests=False, p_values_computed=False,
                  scope="cell-level conditional uncertainty; does not resample animals, trials, bins, fits, or training seeds")
    checks = dict(unique_cells=22, complete_model_pairs=True, exact_source_nll_match=True,
                  exact_cell_group_and_bin_counts=True, parameter_counts_verified_all_cells=True,
                  source_hashes_unchanged=all(sha256(p) == source_hashes[str(p)] for p in sources),
                  training_runs=0, model_loads=0, baseline_changes=0)
    assert checks["source_hashes_unchanged"]
    result = dict(status="COMPLETED", paired=paired, descriptive=descriptive, bootstrap=method, checks=checks,
                  cell_order=[c.cell_id for c in cells], models=MODELS, parameters=parameters)
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = dict(source_sha256=source_hashes, producer_sha256=sha256(Path(__file__)),
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    git_tracked_status=subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=ROOT, text=True),
                    runtime=dict(python=sys.version, numpy=np.__version__), source_contract=preflight["contract"])
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
