# Verification results

Goal/constraints, saved-artifact execution QA, code correctness, file security/provenance and model/input contract reviews: PASS.

- Existing contract tests: 3 passed. All five Python files parse; all four executable analysis modules import. Each file is below 200 non-comment/nonblank lines.
- 22 cells x 4 model/mode labels x 72 stimuli x 150 bins. All 308 saved response tensors finite.
- Original R4 replay: 264 channel tensors bitwise equal to prior results, maximum error zero.
- Sigmoid(logit) versus saved probability: maximum difference 1.49e-8, within float32 tolerance.
- All 1,540 paired-control channel comparisons bitwise equal; AC-off saved currents exact-zero.
- 336 exact-radius histogram checks and all protected target-neighborhood checks passed.
- All 40 annular-statistics rows independently checked: histogram and mean differences zero; std differences at most 2.78e-16 under independent floating-point reduction.
- All 176 checkpoint BC/AC-support statistic rows independently checked: original-to-variant mean, std and histogram differences zero.
- All 76 recorded source/checkpoint hashes match. No model or checkpoint writes, training, optimizer construction or backward passes.
- 4,016 selected per-cell response/comparison mean-on values and all 10,040 group mean-on values independently recomputed: maximum error zero.
- All 704 Mach boundary-extrema rows, including 176 LN rows, independently recomputed exactly.
- The review identified missing dedicated Mach temporal panels. A saved-tensor-only supplement added 27 panels (22 cells, four groups and population), without further inference. Both supplement hashes match; 80 Mach summary rows verified with rounding error at most 4.95e-10.
- 82 PNG figures delivered. Diagnostic population, stimulus and Mach temporal figures visually checked.

In-memory state_dict equality, H1 exact-zero and absent gradients were asserted during inference and recorded in verification.json; the independent saved-artifact reviewer did not rerun these operations.

The variants are custom discrete contour rearrangements motivated by the cited literature, not exact literature displays or verified changes of human perception. Annular matching is target-centered, not centered on each LN's learned off-center Gaussian. The original stimulus and analysis definitions were not changed.
