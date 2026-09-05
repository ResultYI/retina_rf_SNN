# Frozen R4-dev visual-probe model outputs

22 cells: MC ON 5, MC OFF 4, PC ON 9, PC OFF 4. 150 Hz; 17x17 L+M; 100 ms pulse at 300 ms; identical zero observed-spike history.

No training or model changes. All 22 checkpoints/state dictionaries unchanged; H1/AC clamps exact-zero; normal re-entry bitwise equal.

Differences below are mean over 300<=t<400 ms, then equal-cell mean. Positive/negative denotes A-B, not perceived brightness. Probability is per 6.667 ms bin.

Mach numbers are junction response minus matched uniform luminance response (not themselves an overshoot claim). Other pairs: SBC bright-dark surround; Hermann intersection-corridor; White bright-bar target minus dark-bar target.

## Mean-on signatures

| Group | Stimulus/pair | Channel | normal | H1-off | AC-off |
|---|---|---|---:|---:|---:|
| all | Mach dark junction | logit | -0.005075402 | -0.004979360 | +0.000792040 |
| all | Mach dark junction | probability | -0.000735839 | -0.000763187 | -0.000273181 |
| all | Mach bright junction | logit | +0.005075400 | +0.004979362 | -0.000792038 |
| all | Mach bright junction | probability | +0.000708926 | +0.000699739 | -0.000292347 |
| all | SBC | logit | -0.052870687 | -0.051881920 | -0.005385967 |
| all | SBC | probability | -0.006359025 | -0.006382262 | -0.000533850 |
| all | Hermann | logit | -0.026428781 | -0.025943106 | -0.001865509 |
| all | Hermann | probability | -0.004793443 | -0.004739322 | -0.000437284 |
| all | White | logit | +0.026723584 | +0.025955188 | +0.002723859 |
| all | White | probability | +0.003207790 | +0.003185773 | +0.000269432 |
| MC_ON | Mach dark junction | logit | -0.009019857 | -0.006915466 | +0.013154872 |
| MC_ON | Mach dark junction | probability | -0.001423520 | -0.001093177 | +0.001729717 |
| MC_ON | Mach bright junction | logit | +0.009019852 | +0.006915466 | -0.013154876 |
| MC_ON | Mach bright junction | probability | +0.001418108 | +0.001092554 | -0.002365984 |
| MC_ON | SBC | logit | -0.321099973 | -0.304770005 | -0.035627934 |
| MC_ON | SBC | probability | -0.050234557 | -0.047682167 | -0.005583932 |
| MC_ON | Hermann | logit | -0.158503407 | -0.152237377 | -0.012136073 |
| MC_ON | Hermann | probability | -0.027544867 | -0.026580945 | -0.002269821 |
| MC_ON | White | logit | +0.162034869 | +0.152489889 | +0.017760983 |
| MC_ON | White | probability | +0.025336962 | +0.023845395 | +0.002783632 |
| MC_OFF | Mach dark junction | logit | +0.002507193 | -0.000688805 | -0.015978151 |
| MC_OFF | Mach dark junction | probability | +0.000472151 | -0.000080547 | -0.003186289 |
| MC_OFF | Mach bright junction | logit | -0.002507194 | +0.000688806 | +0.015978157 |
| MC_OFF | Mach bright junction | probability | -0.000530398 | -0.000077985 | +0.002275852 |
| MC_OFF | SBC | logit | +0.272373859 | +0.248843882 | +0.036500100 |
| MC_OFF | SBC | probability | +0.047095094 | +0.043396413 | +0.005863975 |
| MC_OFF | Hermann | logit | +0.132665882 | +0.124179281 | +0.012419373 |
| MC_OFF | Hermann | probability | +0.019889679 | +0.018665604 | +0.001567627 |
| MC_OFF | White | logit | -0.137165587 | -0.124495858 | -0.018188455 |
| MC_OFF | White | probability | -0.023733450 | -0.021732640 | -0.002922144 |
| PC_ON | Mach dark junction | logit | -0.008088546 | -0.007153735 | +0.006485149 |
| PC_ON | Mach dark junction | probability | -0.001187654 | -0.001089551 | +0.000755076 |
| PC_ON | Mach bright junction | logit | +0.008088549 | +0.007153741 | -0.006485150 |
| PC_ON | Mach bright junction | probability | +0.001216517 | +0.001082920 | -0.001058683 |
| PC_ON | SBC | logit | -0.107780594 | -0.100598750 | -0.015621133 |
| PC_ON | SBC | probability | -0.015485854 | -0.014585515 | -0.002122810 |
| PC_ON | Hermann | logit | -0.052963434 | -0.050229893 | -0.005431315 |
| PC_ON | Hermann | probability | -0.008190753 | -0.007797045 | -0.000890144 |
| PC_ON | White | logit | +0.054380102 | +0.050311801 | +0.007938124 |
| PC_ON | White | probability | +0.007812156 | +0.007293882 | +0.001078736 |
| PC_OFF | Mach dark junction | logit | -0.000947851 | -0.001957436 | -0.010700807 |
| PC_OFF | Mach dark junction | probability | -0.000067646 | -0.000299022 | -0.002177271 |
| PC_OFF | Mach bright junction | logit | +0.000947843 | +0.001957435 | +0.010700815 |
| PC_OFF | Mach bright junction | probability | -0.000080307 | +0.000124287 | +0.001455757 |
| PC_OFF | SBC | logit | +0.080718667 | +0.073115253 | +0.013559550 |
| PC_OFF | SBC | probability | +0.015566637 | +0.013921263 | +0.002956090 |
| PC_OFF | Hermann | logit | +0.039272808 | +0.036447617 | +0.004710878 |
| PC_OFF | Hermann | probability | +0.006606664 | +0.006037656 | +0.000867408 |
| PC_OFF | White | logit | -0.040753515 | -0.036564518 | -0.006892325 |
| PC_OFF | White | probability | -0.007872257 | -0.006963587 | -0.001502677 |

## Boundary excursions

Computed in fixed x=-6..-2 / +2..+6-pixel regions against both remote plateau mean-on responses. Positive/negative excursions use response units, not an illusion score.

| Mode | Profile | Channel | Cells with above-plateau excursion | Cells with below-plateau excursion | Mean cell max above | Mean cell min below |
|---|---|---|---:|---:|---:|---:|
| normal | ramp | logit | 22/22 | 22/22 | 0.007144058 | -0.007144085 |
| normal | ramp | probability | 22/22 | 22/22 | 0.001224760 | -0.001093031 |
| normal | matched_uniform | logit | 0/22 | 0/22 | 0.000000000 | 0.000000000 |
| normal | matched_uniform | probability | 0/22 | 0/22 | 0.000000000 | 0.000000000 |
| H1_off | ramp | logit | 21/22 | 21/22 | 0.006068373 | -0.006068441 |
| H1_off | ramp | probability | 21/22 | 21/22 | 0.001053025 | -0.000927760 |
| H1_off | matched_uniform | logit | 0/22 | 0/22 | 0.000000000 | 0.000000000 |
| H1_off | matched_uniform | probability | 0/22 | 0/22 | 0.000000000 | 0.000000000 |
| AC_off | ramp | logit | 22/22 | 22/22 | 0.001706188 | -0.001706210 |
| AC_off | ramp | probability | 22/22 | 22/22 | 0.000317758 | -0.000223244 |
| AC_off | matched_uniform | logit | 0/22 | 0/22 | 0.000000000 | 0.000000000 |
| AC_off | matched_uniform | probability | 0/22 | 0/22 | 0.000000000 | 0.000000000 |

## Controls and artifacts

SBC, Hermann and White control A-B traces are bitwise zero for every cell and mode. Mach uniform-control responses and their plateau excursions are reported separately.

Per-cell signed mean, signed/absolute peak, peak time, onset/offset, integral and clamp-minus-normal: per-cell-responses.csv. Group means and direction counts: group-responses.csv. Full raw logits/probabilities and AC currents: response-tensors.pt. All individual spatial profiles/time courses: figures/cell-*.png.

Stimulus definitions and matching boundaries: stimulus-contract.json and stimuli-and-controls.png. Reproducibility and immutable-source checks: verification.json.
