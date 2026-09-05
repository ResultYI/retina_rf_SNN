# Schottdorf–Lee 150 Hz multi-spike-bin sanity

STATUS: PASS

Lineage: `D:\PythonProject\retina_rf_SNN\output\real_data\schottdorf_canonical_v1_shared_bc_development_22cell_20260830`

22 cells / 37 recordings; dt=1000/150 ms. No model/checkpoint loading or training.

All denominators use the actual loss mask. Train is the final full-train refit split, not inner-train; validation is original held-out validation. Each observed trial/bin is counted once, not once per optimizer visit. Combined concatenates these disjoint splits.

Per recording/trial: train segments 0–15, validation 16–19, 150 bins/segment. Each segment uses bins 30–149 (120 bins). Excluded warmup bins are absent from all three ratios.

Ratios: A=count>=2 bins/all valid bins; B=count>=2 bins/nonzero valid bins; C=sum(max(count-1,0))/sum(count). Population/group ratios pool integer numerators and denominators, not per-cell percentage averages.

| Split | All bins | Nonzero | Count>=2 | Spikes | Excess | A | B | C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 263040 | 59728 | 9960 | 71101 | 11373 | 3.786496% | 16.675596% | 15.995556% |
| validation | 65760 | 13946 | 2237 | 16503 | 2557 | 3.401764% | 16.040442% | 15.494153% |
| combined | 328800 | 73674 | 12197 | 87604 | 13930 | 3.709550% | 16.555366% | 15.901100% |

| Group | Cells | Split | A | B | C |
|---|---:|---|---:|---:|---:|
| MC ON | 5 | train | 7.751116% | 34.108210% | 30.100698% |
| MC ON | 5 | validation | 6.927083% | 34.115060% | 30.258114% |
| MC ON | 5 | combined | 7.586310% | 34.109461% | 30.129494% |
| MC OFF | 4 | train | 5.145089% | 21.244240% | 18.944157% |
| MC OFF | 4 | validation | 4.523810% | 19.830398% | 17.779566% |
| MC OFF | 4 | combined | 5.020833% | 20.974761% | 18.724737% |
| PC ON | 9 | train | 1.697917% | 7.756057% | 7.269451% |
| PC ON | 9 | validation | 1.524306% | 7.661431% | 7.221503% |
| PC ON | 9 | combined | 1.663194% | 7.738538% | 7.260578% |
| PC OFF | 4 | train | 2.656250% | 11.550906% | 10.536472% |
| PC OFF | 4 | validation | 2.569444% | 10.698059% | 9.832402% |
| PC OFF | 4 | combined | 2.638889% | 11.374327% | 10.391601% |

Target consistency: 328800 valid bins checked; 0 mismatches. Raw integer counts independently accumulated with numpy.add.at equal production spike_counts. All 22 saved validation target/mask/source IDs/trial orders match exactly.

Training targets/masks are reconstructed from original raw files using unchanged, final-manifest-hash-matched production code; no separately saved training-target tensor is claimed.

Source locations: data/schottdorf_lee_multirecording.py:174–226 (integer binning, split and target/mask); training/mechanistic_retina/r4_development.py:86–87,120–131 (loss and full-train refit); training/mechanistic_retina/losses.py:17–28 (masked Bernoulli NLL); evaluation/mechanistic_retina/karamanlis_prediction_baselines.py:169–187 (validation target/mask). Exact source SHA256 and raw file provenance are in production_target_consistency.json.

Files: per_cell.csv (66 rows), group_summary.csv (12 rows), production_target_consistency.json, and compute.py (reproducible count-only calculation). No data, targets, masks, or production source files were changed.
