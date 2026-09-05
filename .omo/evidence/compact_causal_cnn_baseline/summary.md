# Compact causal CNN: final 22-cell comparison

STATUS: COMPLETED

| Group | Cells | CNN | Constant | Center-surround LN | SC-adapted | Canonical V1 |
|---|---:|---:|---:|---:|---:|---:|
| overall | 22 | 0.422597810626 | 0.509817265651 | 0.425997944041 | 0.458313940340 | 0.438956145536 |
| MC_ON | 5 | 0.372717887163 | 0.496057206392 | 0.395894259214 | 0.422290021842 | 0.428506755829 |
| MC_OFF | 4 | 0.414419904351 | 0.533821441233 | 0.430339150131 | 0.503657426318 | 0.441669881344 |
| PC_ON | 9 | 0.419964339998 | 0.494986361927 | 0.418741527531 | 0.428669068882 | 0.427831285530 |
| PC_OFF | 4 | 0.499050930142 | 0.536382697523 | 0.475613281131 | 0.524701313263 | 0.474335081875 |

## Per cell

| Cell | Group | CNN | Constant | LN | SC-adapted | Canonical V1 | LR | Best / stop |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 67#4 | PC_OFF | 0.498056352139 | 0.522695422173 | 0.479695677757 | 0.528139314136 | 0.464896202087 | 0.0003 | 179 / 379 |
| 67#6 | MC_OFF | 0.401670396328 | 0.522224366665 | 0.430726110935 | 0.488901451752 | 0.442264437675 | 0.001 | 290 / 490 |
| 67#7 | MC_ON | 0.386178344488 | 0.529005765915 | 0.428921610117 | 0.446571966286 | 0.456886798143 | 0.001 | 110 / 310 |
| 67#14 | PC_ON | 0.348917901516 | 0.510558128357 | 0.366128653288 | 0.376259271923 | 0.372924745083 | 0.001 | 216 / 416 |
| 67#21 | PC_ON | 0.306612759829 | 0.370041042566 | 0.298146069050 | 0.304101728669 | 0.309125483036 | 0.0003 | 508 / 708 |
| 67#26 | PC_ON | 0.506272435188 | 0.589472651482 | 0.515616238117 | 0.525191626746 | 0.512515842915 | 0.0003 | 372 / 572 |
| 67#33 | MC_OFF | 0.374034106731 | 0.496288686991 | 0.373482376337 | 0.466717875721 | 0.369917273521 | 0.001 | 214 / 414 |
| 67#34 | PC_ON | 0.366448670626 | 0.484608381987 | 0.377446562052 | 0.396439078905 | 0.386381626129 | 0.001 | 144 / 344 |
| 68#3 | MC_OFF | 0.469915121794 | 0.605145573616 | 0.486940622330 | 0.573757854869 | 0.519419848919 | 0.001 | 282 / 482 |
| 68#4 | PC_ON | 0.444469183683 | 0.458584100008 | 0.423185050488 | 0.437613226712 | 0.427100300789 | 0.001 | 73 / 273 |
| 68#7 | PC_ON | 0.352869927883 | 0.473114222288 | 0.356706738472 | 0.372152367112 | 0.385217726231 | 0.001 | 312 / 512 |
| 68#10 | MC_ON | 0.312622427940 | 0.449517458677 | 0.315079480410 | 0.361510521586 | 0.421071052551 | 0.001 | 96 / 296 |
| 68#11 | PC_OFF | 0.413932442665 | 0.459219723940 | 0.387451291084 | 0.438922587840 | 0.372821658850 | 0.001 | 207 / 407 |
| 69#3 | PC_OFF | 0.579556822777 | 0.632579505444 | 0.563168525696 | 0.618071202765 | 0.553157746792 | 0.001 | 121 / 321 |
| 69#4 | MC_ON | 0.371905148029 | 0.513997793198 | 0.427978813648 | 0.453406195194 | 0.433251053095 | 0.0003 | 174 / 374 |
| 69#6 | MC_OFF | 0.412059992552 | 0.511627137661 | 0.430207490921 | 0.485252522932 | 0.435077965260 | 0.001 | 130 / 330 |
| 69#7 | MC_ON | 0.451202601194 | 0.521963000298 | 0.458073705435 | 0.465085161530 | 0.463302940130 | 0.0003 | 142 / 342 |
| 69#21 | PC_OFF | 0.504658102989 | 0.531036138535 | 0.472137629986 | 0.513672148312 | 0.506464719772 | 0.0003 | 296 / 496 |
| 70#1 | PC_ON | 0.535341799259 | 0.555593311787 | 0.525036752224 | 0.529321234078 | 0.545601189137 | 0.001 | 48 / 248 |
| 70#7 | PC_ON | 0.499858021736 | 0.546998739243 | 0.476247549057 | 0.482691464473 | 0.472401469946 | 0.001 | 89 / 289 |
| 70#15 | PC_ON | 0.418888360262 | 0.465906679630 | 0.430160135031 | 0.434251621322 | 0.439213186502 | 0.001 | 67 / 267 |
| 70#34 | MC_ON | 0.341680914164 | 0.465802013874 | 0.349417686462 | 0.384876264615 | 0.368021935225 | 0.001 | 137 / 337 |

## Fixed architecture and training contract

Conv3D 1->4 (12,5,5), ReLU; Conv3D 4->4 (9,3,3), temporal dilation 6, ReLU; learned 4x11x11 readout.
Temporal left pads 11/48; no spatial padding; spatial sizes 17->13->11. Stimulus receptive field 60 bins (lags 0..59).
LN head: z=readout+history_weight*strictly-past fixed history+bias. No additional output nonlinearity before Bernoulli logits.
Adam, batch 8, seed 61001, lr candidates {0.001,0.0003}; no regularizer, no weight decay, no architecture sweep.
Exact LN inner split and guard; max 1000, patience 200, min_delta 1e-7, dev evaluation every step; raw step zero eligible.
Select lowest unpenalized inner-dev NLL, then fresh full-train refit for best-step count. Original validation never selects a model.
Input tensors generated with the frozen native loader and verified against the corrected SC preflight. GPU training is float32 without AMP or TF32.
Final NLL uses the original native CPU evaluation runtime and identical saved target/mask/order, averaged equally over cells.

## Parameter accounting

CNN: 1204 first-conv + 1300 second-conv + 484 readout + 2 bias/history = 2990 per cell; all trainable and optimizer-listed; 65780 across 22 independent fits.
Constant: 1/cell. LN: 128/cell. SC-adapted: 64 inherited center coordinates + 4 fitted output parameters (68 raw bookkeeping, not functional DoF). Canonical V1: 129 total / 33 trainable per cell.
Prediction/capacity comparison only; no matched-capacity claim. Existing baseline artifacts were not modified or refitted.

## Evidence

preflight.json: frozen definitions, tensor/source hashes, splits, comparison source values.
runtime.json: GPU arithmetic/runtime identity. training_complete.json: 44 inner fits and 22 refits.
cells/*: two inner checkpoints and full dev trajectories, raw/refit checkpoints, validation logits, fitting parameter counts.
per_cell.csv, group_summary.csv, results.json: full-precision comparison results. verification.md: focused contract checks.
