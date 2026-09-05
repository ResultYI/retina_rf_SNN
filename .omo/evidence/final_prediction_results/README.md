# Final frozen prediction results

STATUS: COMPLETED. All prediction baselines remain frozen. This package performs only arithmetic and plotting on existing per-cell artifacts: zero training, zero checkpoint/model loads, zero hyperparameter selection, and zero baseline changes.

## Main deliverables

- [Final main tables](tables.md): overall NLL, three paired comparisons, four descriptive groups, parameter accounting, and all 22 per-cell NLLs for Constant / LN / CNN / Canonical V1.
- [Paper Results draft and figure captions](results_draft.md).
- [Figure 1: paired predictions and mean-difference CIs](figures/main_prediction_paired.png), [vector PDF](figures/main_prediction_paired.pdf), [editable SVG](figures/main_prediction_paired.svg).
- [All 22 per-cell paired differences](figures/per_cell_paired_differences.png), [vector PDF](figures/per_cell_paired_differences.pdf), [editable SVG](figures/per_cell_paired_differences.svg).

## Supplement

- [SC-adapted supplementary tables](supplement.md), including the five-model per-cell comparison.
- [SC-adapted supplementary figure](figures/supplement_sc_adapted.png), [vector PDF](figures/supplement_sc_adapted.pdf), [editable SVG](figures/supplement_sc_adapted.svg).
- [SC-adapted parameter accounting](supplement_sc_parameters.csv): 64 inherited center coordinates + 4 fitted output parameters; no spike-history term.

## Full-precision data and provenance

- `per_cell_nll.csv`: all five models, all 22 cell identities/types, valid train/validation bin counts, and the three paired differences. Values are copied exactly from existing artifacts without rounding.
- `paired_comparisons.csv`: mean, median, both percentile 95% CIs, strict first/second-model win counts, and exact ties.
- `group_descriptive.csv`: overall and four groups, with mean, median, sample SD, quartiles and range. No group significance tests.
- `main_parameters.csv`: CNN 2990 and Canonical V1 33 trainable/optimizer-listed scalar parameters per cell; Canonical V1 has 129 registered parameter scalars including frozen ones. Counts are not independent functional DoF or a matched-capacity claim.
- `analysis.json`: statistics, exact method definition, cell ordering, and consistency checks.
- `bootstrap_resamples.npz`: actual 100000 × 22 paired cell-index draws, zero-based into the saved cell order.
- `source_manifest.json`: SHA256 of all 29 consumed source artifacts, current git HEAD/status, runtime identity, and inherited frozen contract.
- [Verification record](verification.md).

## Locked analysis convention

NLL is in nats per scored Bernoulli bin; lower is better. Population summaries weight each of the 22 cells equally, not by trial length or valid-bin count. Differences are first model minus second model. Resample complete cell pairs with replacement, 22 draws per replicate, 100,000 replicates, NumPy PCG64 seed 20260831; use identical draws for all comparisons. CIs use 2.5/97.5 percentile endpoints with linear interpolation, separately for mean and median differences. No stratification, p-values, multiplicity adjustment, or group tests. Cell-level CIs condition on these frozen fits; they do not incorporate animal-level clustering, between-cell dependence, or training-seed uncertainty.

## Source selection

The CNN package's native-CPU per-cell NLL table is the merge index. Each value is checked exactly against the matching per-cell entry in the frozen LN, final shared-BC Canonical V1, and corrected SC-adapted result artifacts. The old R4 comparator embedded in historical LN results is **not** used as Canonical V1. SC-adapted uses the corrected `69#6` result with official `a0=4`; no fit was rerun here. Cell IDs, cell groups and train/validation valid-bin counts agree across all sources. The existing CNN preflight records the matching input/target/mask identities; this task does not re-execute a loader or evaluation model.

## Reproduction

Use the saved `build_package.py`, `check_statistics.py`, and `render_figures.py` with the recorded local Python runtime. They only read result CSV/JSON files and write within this directory. The bundled runtime lacked matplotlib; the existing project runtime was used without installing packages: Python 3.12.7, NumPy 2.2.6, matplotlib 3.10.8, Pydantic 2.8.2.
