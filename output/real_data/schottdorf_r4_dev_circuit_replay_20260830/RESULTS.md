# R4-dev 22-cell circuit replay

22/22 cells completed. No training, optimizer creation, model edit, or learned-parameter update. All 44 input checkpoints and the recorded source files passed their before/after SHA256 checks. H1/BC/AC structural current clamps were exact-zero. All seven temporal probe input tensors and spatial supports matched the old experiment exactly.

All population values below are equal-cell means. Old = historical 50-step R4; new = R4-dev full-refit checkpoints. RF remains the old endpoint-Jacobian definition, averaged across held-out sequences, retaining only 16 lags (106.6667 ms); this is not a full-duration dynamic RF. Validation perturbation means retain the original all-sequence-bin aggregation, including warmup. Temporal response is relative to the same-clamp zero-input baseline; peak is the signed maximum-absolute response, and latency is time from probe start.

## RF

| RF | Old norm | New norm | Mean old/new cosine |
|---|---:|---:|---:|
| Global | 0.340150 | 0.246663 | 0.901943 |
| H1 | 0.000622 | 0.005646 | 0.799522 |
| BC | 0.284542 | 0.220309 | 0.894380 |
| AC | 0.170421 | 0.098581 | 0.975157 |
| Temporal | 0.375521 | 0.467728 | 0.827839 |

All global/H1/BC/AC RF norms remain nonzero in 22/22 cells. Stored H1+BC+AC decomposition matches global RF numerically.

## Held-out structural perturbations

| Clamp | Old mean absolute logit change | New | Old mean absolute probability change | New |
|---|---:|---:|---:|---:|
| H1-off | 0.009391 | 0.068427 | 0.001143 | 0.008130 |
| BC-off | 1.834241 | 1.055895 | 0.202264 | 0.135562 |
| AC-off | 1.621450 | 0.773259 | 0.196937 | 0.100845 |

H1-off absolute logit effect increased in 19/22 cells. BC-off and AC-off decreased in 22/22 each. No cell reversed its mean signed logit effect for any of the three clamps. H1-off remains smaller than both BC-off and AC-off in all 22 cells. BC-off exceeds AC-off in 17/22 old versus 19/22 new cells; the rank changes occur at 67#7 and 70#34.

## Temporal center-surround

| Normal condition | Old probability peak | New | Old peak time (ms) | New |
|---|---:|---:|---:|---:|
| Center only | 0.116809 | 0.062580 | 403.333 | 410.303 |
| Surround only | -0.073885 | -0.041712 | 412.121 | 416.364 |
| Simultaneous | 0.013663 | 0.018790 | 378.182 | 369.394 |

### Retained numerical patterns

- Normal center-only peaks remain positive in 22/22 cells; surround-only peaks remain negative in 22/22.
- At every offset (-100, -50, 0, +50, +100 ms), normal paired-input probability integral remains below the same cell's center-only integral: 22/22 at each offset.
- AC-off reduces the magnitude of the simultaneous-versus-center integral difference in all 22 cells. Mean per-cell residual fraction: 0.007227 old to 0.045576 new. The residual remains negative, not zero.
- Mean signed AC-off logit change remains positive for both ON groups and negative for both OFF groups; BC-off retains the opposite group directions.

### Changed / no longer universal

- H1 effects increase while BC/AC effects decrease, as quantified above.
- Population signed BC-off probability change reverses: +0.032207362 to -0.004905250. Its per-cell mean signed logit directions do not reverse.
- Mean normal simultaneous-versus-center integral difference: -0.010562297 to -0.005127711 probability-seconds.
- The old all-positive normal peak pattern at both +/-100 ms offsets is no longer universal: 22/22 to 20/22. Cells 68#4 and 69#4 become negative at the signed maximum-absolute peak.
- Cell 69#4 also changes positive to negative at +50 ms. Cell 70#34 changes negative to positive at -50 ms and simultaneous onset.
- These statements concern stored model outputs only.

## Numerical verification boundary

Historical artifacts were not bitwise reproduced: maximum discrepancy across cells was 1.90735e-6 for validation perturbation tensors, 1.49012e-8 for RF tensors, and 2.38419e-7 for temporal response tensors. Strict rtol=1e-6/atol=1e-7 pass/fail flags are preserved per cell; failed flags were not converted to passes. The source of last-bit differences is UNVERIFIED. Old checkpoints were also replayed in the current runtime and saved separately. Model/source files were not changed to eliminate these differences.

Existing temporal-probe tests: 3 passed. Earlier combined check: 8 passed, 1 pre-existing missing-document failure (`docs/mathematical_formulation/retina_rf_snn_formulas_zh.tex`); the missing document was not recreated.

## Artifacts

- `replay-results.json`: contracts, SHA256 source manifest, per-cell checks, 462 temporal metric rows.
- `comparison.json`: population/four-group comparisons and sign counts.
- `rf-tensors.pt`: global, temporal, H1, BC, AC RFs for 22 cells.
- `perturbation-tensors.pt`: H1/BC/AC-off response deltas and clamped RFs.
- `validation-normal-responses.pt`: normal held-out logit/probability traces.
- `response_tensors.pt`: seven temporal probe conditions, normal/H1-off/AC-off responses and inputs.
- `old-checkpoint-replayed.pt`: old-checkpoint current-runtime RF, perturbation and temporal tensors.
- `rf-comparison.csv`, `rf-group-comparison.csv`: per-cell and population/four-group RF comparisons.
- `perturbation-comparison.csv`, `perturbation-group-comparison.csv`: per-cell and population/four-group structural effects.
- `temporal-comparison.csv`, `temporal-group-comparison.csv`: old/new/change values for all original temporal metrics.
- `per_cell_metrics.csv`, `group_summary.csv`: new temporal metrics in the existing report schema.
- `figures/cells/`, `figures/groups/`: 22 cell and four group temporal-response figures.
