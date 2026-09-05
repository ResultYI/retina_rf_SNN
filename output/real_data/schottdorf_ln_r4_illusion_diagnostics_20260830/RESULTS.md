# Frozen LN versus R4-dev illusion diagnostics

22 cells: MC ON 5, MC OFF 4, PC ON 9, PC OFF 4. Original 63 input sequences, timing, controls, history and metrics reused unchanged. Diagnostic views use the same envelope and 150 Hz time axis. No training/model edits.

Original R4 replay max absolute logit/probability error: 0. All learned states unchanged; H1/AC structural clamps exact-zero. Target-centered exact-radius luminance histograms are preserved separately for each target; this does not assert matching around learned off-center LN Gaussian centers.

White custom remote-contour rearrangement is motivated by [Howe 2001](https://doi.org/10.1068/p3212) / [Howe 2005](https://doi.org/10.1068/p5414). Hermann custom contour rearrangement is motivated by [Geier et al. 2008](https://doi.org/10.1068/p5622). The 17x17 permutation produces fragmented contours, not a smooth curved grid. These are discrete diagnostic constructions, not exact reproductions or claims of a particular perceptual effect.

Construction and all fixed constants: protocol.json. Original and diagnostic are separately centered local views. For every exact radius, angle-ordered pixel luminances are cyclically permuted; protected radii White=4 px and Hermann=2 px retain all target pixels. No response-dependent stimulus selection.

## Mean-on response differences

Mean over 300<=t<400 ms, then equal-cell mean. Mach entries are junction minus uniform-control responses. SBC is bright-surround minus dark-surround; Hermann is intersection minus corridor; White is bright-bar target minus dark-bar target. Signed response units, no perceptual score.

| Group | Pair | Channel | LN | R4 normal | R4 H1-off | R4 AC-off |
|---|---|---|---:|---:|---:|---:|
| all | Mach dark | logit | -0.011784341 | -0.005075402 | -0.004979360 | +0.000792040 |
| all | Mach dark | probability | -0.002153544 | -0.000735839 | -0.000763187 | -0.000273181 |
| all | Mach bright | logit | +0.008500861 | +0.005075400 | +0.004979362 | -0.000792038 |
| all | Mach bright | probability | +0.001219089 | +0.000708926 | +0.000699739 | -0.000292347 |
| all | SBC | logit | -0.044822165 | -0.052870687 | -0.051881920 | -0.005385967 |
| all | SBC | probability | -0.007585129 | -0.006359025 | -0.006382262 | -0.000533850 |
| all | Hermann original | logit | -0.005099510 | -0.026428781 | -0.025943106 | -0.001865509 |
| all | Hermann original | probability | -0.000466089 | -0.004793443 | -0.004739322 | -0.000437284 |
| all | Hermann diagnostic | logit | -0.013071541 | -0.026426589 | -0.025943106 | -0.001860964 |
| all | Hermann diagnostic | probability | -0.001739710 | -0.004793384 | -0.004739322 | -0.000436384 |
| all | White original | logit | +0.011775351 | +0.026723584 | +0.025955188 | +0.002723859 |
| all | White original | probability | +0.002532775 | +0.003207790 | +0.003185773 | +0.000269432 |
| all | White diagnostic | logit | +0.020605929 | +0.026736152 | +0.025955188 | +0.002739225 |
| all | White diagnostic | probability | +0.003689361 | +0.003208899 | +0.003185773 | +0.000270893 |
| MC_ON | Mach dark | logit | +0.006503557 | -0.009019857 | -0.006915466 | +0.013154872 |
| MC_ON | Mach dark | probability | +0.000533694 | -0.001423520 | -0.001093177 | +0.001729717 |
| MC_ON | Mach bright | logit | -0.019254610 | +0.009019852 | +0.006915466 | -0.013154876 |
| MC_ON | Mach bright | probability | -0.001677193 | +0.001418108 | +0.001092554 | -0.002365984 |
| MC_ON | SBC | logit | +0.060088020 | -0.321099973 | -0.304770005 | -0.035627934 |
| MC_ON | SBC | probability | +0.005367424 | -0.050234557 | -0.047682167 | -0.005583932 |
| MC_ON | Hermann original | logit | +0.008461893 | -0.158503407 | -0.152237377 | -0.012136073 |
| MC_ON | Hermann original | probability | +0.001136943 | -0.027544867 | -0.026580945 | -0.002269821 |
| MC_ON | Hermann diagnostic | logit | +0.014354639 | -0.158494705 | -0.152237377 | -0.012115161 |
| MC_ON | Hermann diagnostic | probability | +0.001480497 | -0.027543835 | -0.026580945 | -0.002265980 |
| MC_ON | White original | logit | -0.025941478 | +0.162034869 | +0.152489889 | +0.017760983 |
| MC_ON | White original | probability | -0.002851091 | +0.025336962 | +0.023845395 | +0.002783632 |
| MC_ON | White diagnostic | logit | -0.023626350 | +0.162094051 | +0.152489889 | +0.017831001 |
| MC_ON | White diagnostic | probability | -0.002646237 | +0.025346231 | +0.023845395 | +0.002794606 |
| MC_OFF | Mach dark | logit | -0.011669419 | +0.002507193 | -0.000688805 | -0.015978151 |
| MC_OFF | Mach dark | probability | -0.001584586 | +0.000472151 | -0.000080547 | -0.003186289 |
| MC_OFF | Mach bright | logit | +0.059422197 | -0.002507194 | +0.000688806 | +0.015978157 |
| MC_OFF | Mach bright | probability | +0.007779378 | -0.000530398 | -0.000077985 | +0.002275852 |
| MC_OFF | SBC | logit | -0.242409706 | +0.272373859 | +0.248843882 | +0.036500100 |
| MC_OFF | SBC | probability | -0.032896859 | +0.047095094 | +0.043396413 | +0.005863975 |
| MC_OFF | Hermann original | logit | -0.079710180 | +0.132665882 | +0.124179281 | +0.012419373 |
| MC_OFF | Hermann original | probability | -0.008794358 | +0.019889679 | +0.018665604 | +0.001567627 |
| MC_OFF | Hermann diagnostic | logit | -0.085706733 | +0.132653011 | +0.124179281 | +0.012398157 |
| MC_OFF | Hermann diagnostic | probability | -0.009788752 | +0.019887534 | +0.018665604 | +0.001564867 |
| MC_OFF | White original | logit | +0.127807949 | -0.137165587 | -0.124495858 | -0.018188455 |
| MC_OFF | White original | probability | +0.018526379 | -0.023733450 | -0.021732640 | -0.002922144 |
| MC_OFF | White diagnostic | logit | +0.135278383 | -0.137228975 | -0.124495858 | -0.018259454 |
| MC_OFF | White diagnostic | probability | +0.019666250 | -0.023743601 | -0.021732640 | -0.002933563 |
| PC_ON | Mach dark | logit | +0.006761310 | -0.008088546 | -0.007153735 | +0.006485149 |
| PC_ON | Mach dark | probability | +0.000631529 | -0.001187654 | -0.001089551 | +0.000755076 |
| PC_ON | Mach bright | logit | -0.006997340 | +0.008088549 | +0.007153741 | -0.006485150 |
| PC_ON | Mach bright | probability | -0.000777833 | +0.001216517 | +0.001082920 | -0.001058683 |
| PC_ON | SBC | logit | +0.081573864 | -0.107780594 | -0.100598750 | -0.015621133 |
| PC_ON | SBC | probability | +0.009089937 | -0.015485854 | -0.014585515 | -0.002122810 |
| PC_ON | Hermann original | logit | +0.043510794 | -0.052963434 | -0.050229893 | -0.005431315 |
| PC_ON | Hermann original | probability | +0.005361207 | -0.008190753 | -0.007797045 | -0.000890144 |
| PC_ON | Hermann diagnostic | logit | +0.033846257 | -0.052952823 | -0.050229893 | -0.005416797 |
| PC_ON | Hermann diagnostic | probability | +0.003979203 | -0.008189304 | -0.007797045 | -0.000887797 |
| PC_ON | White original | logit | -0.043761183 | +0.054380102 | +0.050311801 | +0.007938124 |
| PC_ON | White original | probability | -0.004374815 | +0.007812156 | +0.007293882 | +0.001078736 |
| PC_ON | White diagnostic | logit | -0.041413036 | +0.054423432 | +0.050311801 | +0.007987328 |
| PC_ON | White diagnostic | probability | -0.004156733 | +0.007817996 | +0.007293882 | +0.001085425 |
| PC_OFF | Mach dark | logit | -0.076486848 | -0.000947851 | -0.001957436 | -0.010700807 |
| PC_OFF | Mach dark | probability | -0.012347966 | -0.000067646 | -0.000299022 | -0.002177271 |
| PC_OFF | Mach bright | logit | +0.027144814 | +0.000947843 | +0.001957435 | +0.010700815 |
| PC_OFF | Mach bright | probability | +0.002772228 | -0.000080307 | +0.000124287 | +0.001455757 |
| PC_OFF | SBC | logit | -0.262763418 | +0.080718667 | +0.073115253 | +0.013559550 |
| PC_OFF | SBC | probability | -0.035982988 | +0.015566637 | +0.013921263 | +0.002956090 |
| PC_OFF | Hermann original | logit | -0.056813779 | +0.039272808 | +0.036447617 | +0.004710878 |
| PC_OFF | Hermann original | probability | -0.007253025 | +0.006606664 | +0.006037656 | +0.000867408 |
| PC_OFF | Hermann diagnostic | logit | -0.080284118 | +0.039262981 | +0.036447617 | +0.004698285 |
| PC_OFF | Hermann diagnostic | probability | -0.010583479 | +0.006604580 | +0.006037656 | +0.000865040 |
| PC_OFF | White original | logit | +0.067845992 | -0.040753515 | -0.036564518 | -0.006892325 |
| PC_OFF | White original | probability | +0.008811080 | -0.007872257 | -0.006963587 | -0.001502677 |
| PC_OFF | White diagnostic | logit | +0.100766494 | -0.040792474 | -0.036564518 | -0.006935046 |
| PC_OFF | White diagnostic | probability | +0.013285681 | -0.007880738 | -0.006963587 | -0.001511989 |

## Annular matching

Maximum original-to-diagnostic mean luminance change: 0.
Maximum std change: 2.77555756156e-16; maximum sorted histogram error: 0.

Per-ring original/diagnostic means, std, min/max, three luminance fractions, changed-pixel counts: annular-luminance.csv. Actual per-cell BC/AC support statistics: per-cell-spatial-support-statistics.csv.

All original and diagnostic contextual paired controls A-B are exact-zero. Mach uniform-control profile and plateau excursions are in mach-boundary-extrema.csv.

## Files

- per-cell-responses.csv / group-responses.csv: signed mean, peak, latency, integral, onset/offset and direction counts.
- per-cell-comparisons.csv / group-comparisons.csv: diagnostic minus original and each R4 mode minus LN, using identical response metrics.
- responses.pt: raw logit/probability time courses, R4 AC currents, names and cell identities.
- figures/: original and diagnostic time courses for every cell and all four groups; diagnostic-stimuli.png: input views.
- verification.json: frozen checkpoint/source hashes and execution checks.

Mach temporal supplement: `MACH_RESULTS.md`, `figures/mach-time-*.png` (all 22 cells, four groups and population), and `mach-supplement-provenance.json`. Generated from the same saved response tensors without further inference.
