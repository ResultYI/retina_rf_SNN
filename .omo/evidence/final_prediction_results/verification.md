# Final prediction package verification

## Data and scope

- PASS: 22 unique cell IDs, five complete paired model columns; no excluded or imputed cell.
- PASS: MC ON / MC OFF / PC ON / PC OFF counts = 5 / 4 / 9 / 4.
- PASS: NLL values match the existing CNN, LN, final shared-BC Canonical V1 and corrected SC-adapted per-cell artifacts exactly. Cell-group and train/validation valid-bin counts also match.
- PASS: all 22 CNN fit artifacts record 2990 total/requires-grad/optimizer-listed parameters; all 22 Canonical artifacts record 129 total and 33 requires-grad/optimizer-listed parameters. LN per-cell artifacts record 128 parameter scalars.
- PASS: hashes of all 29 consumed source artifacts are unchanged. Existing tracked worktree changes were preserved.
- Model/checkpoint loads = 0; training = 0; hyperparameter searches = 0; baseline modifications = 0.

## Statistical verification

- Failing-first check: the numerical check was invoked before the analysis module existed and failed with the expected missing-file error.
- PASS after implementation: constant paired differences yield collapsed CIs with the expected win count; identical predictions yield zero differences, collapsed zero intervals, and 22 exact ties.
- PASS: exactly 100,000 resamples of 22 cell indices, generated once and shared across all three comparisons; the draws are saved.
- PASS: independent bootstrap mean calculation using per-replicate cell-frequency weights, rather than indexed difference means, agrees with the recorded CIs. Independent sorted-array linear percentile interpolation agrees; maximum mean-CI discrepancy is 8.673617379884035e-19 nats/bin.
- PASS: median CIs agree exactly with independently interpolated sorted bootstrap medians. Means/medians/wins also match standard-library calculations.
- PASS: all 25 group-by-model mean NLL values agree with a separate standard-library aggregation to <1e-15.
- No p-values, small-group inferential tests, adjusted confidence intervals, or model-selection decisions were added.

## Artifact checks

- Three report-only Python files compile and run; scoped programming check: `no violations in 3 file(s)`.
- Three scientific figures rendered in PNG/PDF/SVG. All three PNGs were inspected: titles, legends, labels, identities, CI endpoints, axes, and cell rows are visible; no clipping or missing panel.
- Main figure contains only Constant, LN, CNN, Canonical V1. SC-adapted is isolated in the supplementary figure/text/tables; the merged full-precision CSV retains all five requested model columns.
- Analysis and plotting have separate responsibilities; typed cell parsing is confined to the CSV boundary. No model, optimizer, loader, or training module is imported.
- Scientific table rules kept source values separate from derived statistics, preserved source hashes, and documented units and resampling. Minimal report scripts avoided modifying research code or creating a new benchmark.
