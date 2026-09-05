# Canonical V1 illusion aggregation check

## Frozen evidence contract

`COMPLETE`. The calculation reads only the saved Canonical V1 `normal` and `AC_off` surfaces in `../per_cell_curves.csv`. It does not load a model, train, or generate a probe. Confidence intervals are 100,000 paired-cell bootstrap draws (`seed=20260902`). Raw cohort means resample all 22 cells; polarity-equal means resample ON and OFF separately and weight each by 1/2; four-class-equal means resample MC ON, MC OFF, PC ON, and PC OFF separately and weight each by 1/4.

## Cell labels and effect-sign partition

| Group | Cells |
|---|---:|
| MC ON | 5 |
| MC OFF | 4 |
| PC ON | 9 |
| PC OFF | 4 |
| ON total | 14 |
| OFF total | 8 |

Across all 80 analyzed surface points, the AC-effect sign partition is exact:

- SBC bright-minus-dark and Mach dark-ramp-minus-uniform: all 14 ON cells positive, all 8 OFF cells negative.
- Mach bright-ramp-minus-uniform: all 8 OFF cells positive, all 14 ON cells negative.
- All 66 cell-by-signature surfaces retain their expected sign at every analyzed point.

The exact pointwise and cellwise audits are in `effect_sign_point_audit.csv` and `effect_sign_cell_audit.csv`.

## Normal to AC-off cohort reversal

Counts are reversed-sign points over the analyzed grid. For SBC, the zero-context radius is excluded. For Mach, width 0 is retained as the no-ramp step-edge control.

| Weighting | SBC | Mach dark | Mach bright |
|---|---:|---:|---:|
| Raw 22-cell | 20/20 | 30/30 | 30/30 |
| ON/OFF equal | 20/20 | 0/30 | 5/30 |
| Four-class equal | 20/20 | 0/30 | 5/30 |

The five retained Mach-bright reversals under equal weighting are exactly the five nonzero-contrast width-0 controls. Therefore the width-greater-than-zero Mach cohort reversal depends on the 14:8 ON/OFF composition. The SBC aggregate reversal persists under both equal-weighting schemes, although each of the four group means keeps its own original sign.

At contrast 0.5 and extent/width 8 px:

| Signature | Weighting | Normal | AC-off | AC-off - normal |
|---|---|---:|---:|---:|
| SBC | Raw 22-cell | -0.077750 | 0.006110 | 0.083860 |
| SBC | ON/OFF equal | -0.018949 | 0.012804 | 0.031753 |
| SBC | Four-class equal | -0.032816 | 0.011935 | 0.044751 |
| Mach dark | Raw 22-cell | -0.007416 | 0.005344 | 0.012761 |
| Mach dark | ON/OFF equal | -0.007162 | -0.001964 | 0.005198 |
| Mach dark | Four-class equal | -0.006852 | -0.000595 | 0.006257 |
| Mach bright | Raw 22-cell | 0.007416 | -0.005344 | -0.012761 |
| Mach bright | ON/OFF equal | 0.007162 | 0.001964 | -0.005198 |
| Mach bright | Four-class equal | 0.006852 | 0.000595 | -0.006257 |

Full four-group and aggregation curves with bootstrap intervals are in `group_surface_curves.csv` and `aggregation_surface_curves.csv`; `group_normal_acoff_curves.png` plots the four group curves at contrast 0.5.

## Mach control-subtracted AC interaction

The reported interaction is

`[(AC-off - normal)_width] - [(AC-off - normal)_width0]`.

At contrast 0.5:

| Signature | Width | Raw 22-cell [95% CI] | ON/OFF equal [95% CI] | Four-class equal [95% CI] |
|---|---:|---:|---:|---:|
| Dark | 2 | -0.030126 [-0.057167, -0.001794] | -0.012192 [-0.020229, -0.005130] | -0.013315 [-0.019299, -0.008556] |
| Dark | 4 | -0.055650 [-0.106316, -0.002458] | -0.022207 [-0.038076, -0.007378] | -0.025524 [-0.036216, -0.016743] |
| Dark | 8 | -0.068413 [-0.130705, -0.003110] | -0.027379 [-0.047213, -0.008684] | -0.031762 [-0.044883, -0.020909] |
| Dark | 12 | -0.072667 [-0.138842, -0.003331] | -0.029112 [-0.050292, -0.009136] | -0.033848 [-0.047787, -0.022301] |
| Dark | 16 | -0.074794 [-0.142917, -0.003451] | -0.029978 [-0.051847, -0.009361] | -0.034890 [-0.049232, -0.023004] |
| Bright | 2 | -0.030078 [-0.057042, -0.001839] | -0.012196 [-0.020213, -0.005164] | -0.013303 [-0.019281, -0.008549] |
| Bright | 4 | -0.004554 [-0.009583, 0.000160] | -0.002180 [-0.005303, 0.000690] | -0.001093 [-0.002628, 0.000024] |
| Bright | 8 | 0.008210 [-0.001493, 0.017879] | 0.002992 [-0.002692, 0.008813] | 0.005144 [0.003415, 0.007048] |
| Bright | 12 | 0.012464 [-0.001136, 0.025722] | 0.004725 [-0.002146, 0.011744] | 0.007230 [0.004883, 0.009825] |
| Bright | 16 | 0.014590 [-0.000989, 0.029702] | 0.005591 [-0.001882, 0.013217] | 0.008273 [0.005599, 0.011237] |

Across all five nonzero contrasts, dark-ramp interactions exclude zero at all 25 width/contrast points under all three weighting contracts. Bright-ramp interactions exclude zero at width 2 under all three contracts; under four-class equal weighting they are also positive at widths 8, 12, and 16 (15/15 points), while width 4 is unresolved.

The per-cell direction is polarity-structured. For dark ramps, every MC/PC ON cell has a negative interaction and every MC/PC OFF cell has a positive interaction at all widths. For bright ramps, all ON cells are negative and all OFF cells positive at widths 2 and 4; at widths 8-16 all ON cells become positive, MC OFF cells become negative, and PC OFF is negative in 3/4 cells at widths 8 and 12 and 4/4 at width 16. Exact counts and four-group bootstrap intervals are in `mach_interaction_direction_summary.csv`; full per-cell values are in `mach_control_subtracted_per_cell.csv`.

## Main-text boundary

The raw Mach normal-to-AC-off cohort reversal is composition-dependent and should not be presented as a cohort-universal effect. The supported main-text result is narrower: AC-off changes the Mach width-response relative to the width-0 step-edge control, with a robust dark-ramp interaction across raw, polarity-equal, and four-class-equal weighting, and a polarity/cell-class-dependent bright-ramp interaction. SBC aggregate reversal is descriptively preserved after equal weighting but remains cancellation-derived because no four-class mean reverses individually.

## Evidence index

- Frozen labeled surfaces: `labeled_surfaces.csv`
- Four group curves: `group_surface_curves.csv`
- Raw/polarity-equal/four-class-equal curves: `aggregation_surface_curves.csv`
- Effect-sign audits: `effect_sign_point_audit.csv`, `effect_sign_cell_audit.csv`
- Cohort reversal audit: `cohort_reversal_audit.csv`
- Mach control subtraction and interaction: `mach_control_subtracted_per_cell.csv`, `mach_control_subtracted_groups.csv`, `mach_control_subtracted_aggregations.csv`
- Interaction directions: `mach_interaction_direction_summary.csv`
- Figures: `group_normal_acoff_curves.png`, `mach_ac_interaction_by_group.png`
- Contract and verification: `aggregation_manifest.json`, `conclusion_metrics.json`, `verification.json`
