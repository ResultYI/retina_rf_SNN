# Read-only identifiability audit protocol

Frozen before the primary replay on 2026-09-05. All new files stay in this directory. Training and optimizer steps: 0. Existing checkpoints, source, data and artifacts are immutable.

## Gate 1: mandatory exact primary replay

Use the original fixed-alignment producer's `preflight.factory/load_data`, original movie/spike adapter and `evaluate_retinal_model` (CPU torch 2.6.0, two threads, default chunk size 8). For every one of the 22 final aligned models, require current schema/revision, Canonical V1 name, shared-BC and overlapping-full-disk contracts, final full-train refit steps, 33 trainable scalars, and no original-validation selection. Strict-load without conversion.

Reconstruct original validation from raw movie/spikes. Require exact target, mask, source-image order and trial order; bitwise-equal saved validation logits; exactly equal saved NLL. Check frozen LN-to-degree center identity. Preserve model tensors and parameter gradients. Save raw replay inputs/logits in a new audit artifact to avoid repeated movie decoding in later permitted diagnostics.

If any primary replay check fails, save the failed gate and STOP before mechanism analysis. Do not change tolerances, source, checkpoints or contracts to obtain a pass.

## Subsequent evidence analysis

After Gate 1 only: inspect the RF/forward/parameter definitions and use existing frozen RF, intervention, effective-parameter, synthetic and four-cell seed artifacts. Separate zero-vs-aligned nuisance sensitivity, zero-centered seed sanity, and same-family synthetic recovery; never pool their interpretations. All real intervention vector statistics use the existing validation score mask and off-minus-normal logit convention.

Missing current-contract evidence is UNVERIFIED, not stability. No new seed training, illusion inference, parameter fitting or architecture changes. Additional inference diagnostic details, if required, will be frozen here before that diagnostic runs.

## Frozen inference diagnostics (before execution)

1. For all 22 aligned models, differentiate the final logit through the complete original 150-bin validation sequence, with observed events held fixed. Use the original ordered three-forward decomposition (normal, H1 off, H1 and AC off). Save all context Jacobians, compare the last-16-bin context mean with the saved RF, and compute retained squared-norm energy per context and after pooling energies (never square the context mean to estimate retention).
2. Fixed long-history diagnostic on the same four cells already selected by the independent-seed protocol: 67#4, 67#6, 68#4, 69#4. Concatenate the four consecutive original validation segments of the first recording/trial into 600 bins (4 s); assert contiguous source frame ranges. Reset only at the start of this diagnostic sequence. This is a different conditioning/reset convention, not a replacement production RF. The largest configured single state tau is 250 ms; 4 s covers 16 such time constants, also allowing the finite 100 ms BC kernel and at most 90 ms combined explicit H1/BC/AC delays. Report oldest-100-bin energy as a finite-window residual check, not an assertion of mathematically finite recurrent memory. Do not scan alternative lengths.
3. Strict-load the existing 4 cells x 3 zero-center fits, verify saved target/mask/order against the primary replay data, and calculate their original 16-bin ordered RF and effective temporal values. Primary RF must reproduce its existing artifact. No new aligned seeds are available. Reuse all saved validation/intervention vectors. For synthetic prediction similarity only, strict-load teacher and three trained students and evaluate their saved validation inputs with zero observed history; verify probabilities against saved teacher target and saved float32 KL. Do not run synthetic parameter recovery or any fitting.

## Frozen descriptive statistics

All similarities and moments accumulate in float64. Relative L2 uses fit A norm; magnitude ratio is fit B mean-absolute effect divided by fit A. Cosine is undefined for zero norm. Real intervention/prediction statistics use the identical validation score mask. Off-minus-normal is the signed intervention convention. Sign agreement excludes exact zeros only; report signed means, RMS and full-vector cosine separately. RF energy centers and RMS extents use sum of squared Jacobians over lag, in the same degree coordinate frame. RF temporal marginal is the signed sum over cones; save an energy marginal as well.

Compare every pair among the three existing real fits within each of the four cells (12 pairs), all 22 zero/aligned pairs, and each of the three trained synthetic students against their teacher; keep cohorts separate. Also report all three synthetic raw/teacher RF comparisons to expose shared-family initialization. No threshold declares statistical prediction equivalence. NLL differences, logit RMSE and mechanism distances remain continuous descriptive evidence.

For tau/delay bounds, lower/upper proximity means normalized position <=0.05 or >=0.95; this is descriptive, not significance. Report fixed bounds and active conditional lower bounds imposed by paired ordering. Temporal pair distance is RMS of effective-value differences divided by each fixed bound width, retaining BC mode labels; it is not invariant to exact mode permutations. Seed SD uses sample SD (n=3), CV=SD/absolute mean. Rank correlations use average ranks, undefined for constants. Alignment summaries use per-mode medians and direction counts across 22 cells. Positive effective gains/amplitude and history gate are recorded separately from tau/delay. No new physiological inference is licensed by these summaries.

## Evidence-bound amendment after discovering synthetic source drift

The old N=8 synthetic teacher strict-loads but current full-forward probability differs from the saved target (maximum 0.0119702219963; unchanged by one versus two CPU threads). Its producer hashes differ in seven files subsequently changed by the 2026-08-31 multi-cell correctness patch; the N=1 compatibility guarantee does not cover this N=8 experiment. Do not loosen replay equality, convert checkpoints, or modify production code. Report this historical shared-BC evidence separately from current implementation applicability.

The synthetic artifact contains all four post-gain pathway currents for normal and each clamp. To recover historical prediction similarity without executing obsolete code: sum these saved currents in original forward order and pass through the unchanged RGC module loaded from the original checkpoint, with zero history. Before accepting this reconstruction, require unchanged original producer hashes for rgc_state.py and state.py, exact teacher probabilities, exact stored float32 KL for each student, and exact all three saved off-minus-normal logit deltas for each role. Save reconstructed historical logits and current full-forward mismatch measurements separately. This is archived-current reconstruction, not a passing current full-model replay. If reconstruction fails, leave historical prediction cosine/RMSE UNVERIFIED.
