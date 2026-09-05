# History semantics audit — fixed before runtime comparison
Scope: existing 22 final N=1 checkpoints; saved 72 illusion input traces and 35 pairs.
No training, no optimizer, no parameter change, no new visual probes, no production edits.
Load each saved model_config without overriding architecture_mode; strict-load its state.
Run four existing modes: normal, H1_off, direct_BC_off, AC_off.
For each mode use three external histories shared identically by all 72 stimulus traces:
1. saved_zero: exact saved all-zero history;
2. periodic_11: one event at time-bin indices congruent to 3 modulo 11;
3. dense_one: one event every time bin (bounded binary stress input, not a physiological rate estimate).
All history prefixes start at the same sequence boundary. No self-generated histories are used.
Primary estimands: pointwise paired logit A−B and paired (clamp−normal) difference; mean over saved 300<=t<400ms window.
Secondary: paired probability curves and stimulus-on means, sign categories, and post-onset absolute-peak index; do not assume history invariance. Other existing metrics are covered by the algebraic trace result, not individually numerically reproduced here.
Source algebra, not a finite history sample, determines the all-shared-history assertion.
Floating-point criterion fixed as bound = 32*eps(dtype)*(1+max_abs_zero_logit+max_abs_changed_logit+max_abs_history_term+max_abs_bias).
Primary max absolute residual must not exceed the numerical tolerance above (difference-of-differences: bound_clamp + bound_normal). This is a preset conservative tolerance, not a claimed tight floating-point error theorem.
Upstream currents and state tensors must remain bitwise equal across histories; context-matched controls must remain exactly zero.
Sign flips are reported both under the existing 1e-9 reporting threshold and after excluding values whose magnitudes are within the roundoff bound.
Record peak-time changes separately: argmax is discontinuous near ties.
Repeat the cell with largest saved absolute history coefficient in each of the four type/polarity groups in float64 (sorted-path first breaks ties), with unchanged saved parameters promoted to float64. This metadata-only selection was amended before response comparisons after observing 18 zero gates; it prevents a trivial inactive-history precision check. All 22 primary cells remain included.
One negative control: checkpoint with largest saved absolute history coefficient, normal mode; use zero history for even stimulus indices and dense-one for odd indices. Confirm history cancellation fails when histories differ, and residual agrees with the explicit history-term difference. This is not an alternative inference protocol.
Integrity: SHA256 relevant production/application sources, saved input, all 22 checkpoints and existing frozen replay artifacts before/after; model state unchanged; gradients absent.
