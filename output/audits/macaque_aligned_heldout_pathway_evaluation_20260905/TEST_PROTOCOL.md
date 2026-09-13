# Preregistered held-out natural-movie protocol

Frozen UTC: 2026-09-05T10:13:33.223510+00:00
Branch: rgc-readout-v2; HEAD: fea28de038821fadee279b93728688b34bcb3bac

## Data gate and immutable range

Eligibility passed before this protocol and before any new model output; see DATA_USAGE_REPORT.md and data_usage_gate.json. No result-based range selection. Live [20,60) seconds; live integer frames [3000,9000); decoded integer movie frames [3751,9751), zero-based half-open. Frozen live origin 751 retains its acquisition-provenance UNKNOWN. Movie: data/real/schottdorf_lee_2021_macaque/1x10_256.mpg; raw recordings are the exact paths below, with original hashes in evidence_manifest.initial.json.

## Exact cells, recordings and local trials

| Cell | Recording | Kind | One-based local trials |
|---|---|---|---|
| 67#4 | lSS01071 | 10min | 1 |
| 67#6 | lSS01078 | 6x1min | 1,2,3,4,5,6 |
| 67#6 | lSS01079 | 10min | 1 |
| 67#7 | lSS01086 | 6x1min | 1,2,3,4,5,6 |
| 67#7 | lSS01087 | 10min | 1 |
| 67#14 | lSS01110 | 6x1min | 1,2,3,4,5,6 |
| 67#14 | lSS01112 | 10min | 1 |
| 67#21 | lSS01130 | 6x1min | 1,2,3,4,5,6 |
| 67#21 | lSS01131 | 10min | 1 |
| 67#26 | lSS01141 | 6x1min | 1,2,3,4,5,6 |
| 67#26 | lSS01142 | 10min | 1 |
| 67#33 | lSS01159 | 6x1min | 1,2,3,4,5,6 |
| 67#33 | lSS01160 | 10min | 1 |
| 67#34 | lSS01167 | 6x1min | 1,2,3,4,5,6 |
| 67#34 | lSS01168 | 10min | 1 |
| 68#3 | lSS01181 | 6x1min | 1,2,3,4,5,6 |
| 68#3 | lSS01183 | 10min | 1 |
| 68#4 | lSS01184 | 6x1min | 1,2,3,4,5,6 |
| 68#7 | lSS01194 | 6x1min | 1,2,3,4,5,6 |
| 68#7 | lSS01196 | 10min | 1 |
| 68#10 | lSS01221 | 10min | 1 |
| 68#11 | lSS01225 | 6x1min | 1,2,3,4,5,6 |
| 68#11 | lSS01227 | 10min | 1 |
| 69#3 | lSS01251 | 6x1min | 1,2,3,4,5,6 |
| 69#3 | lSS01252 | 10min | 1 |
| 69#4 | lSS01254 | 6x1min | 1,2,3,4,5,6 |
| 69#6 | lSS01256 | 6x1min | 1,2,3,4,5,6 |
| 69#6 | lSS01257 | 10min | 1 |
| 69#7 | lSS01258 | 6x1min | 1,2,3,4,5,6 |
| 69#7 | lSS01259 | 10min | 1 |
| 69#21 | lSS01270 | 6x1min | 1,2,3,4,5,6 |
| 70#1 | lSS01278 | 6x1min | 1,2,3,4,5,6 |
| 70#7 | lSS01284 | 6x1min | 1,2,3,4,5,6 |
| 70#7 | lSS01285 | 10min | 1 |
| 70#15 | lSS01287 | 6x1min | 1,2,3,4,5,6 |
| 70#34 | lSS01299 | 6x1min | 1,2,3,4,5,6 |
| 70#34 | lSS01300 | 10min | 1 |

Cell order: 67#4, 67#6, 67#7, 67#14, 67#21, 67#26, 67#33, 67#34, 68#3, 68#4, 68#7, 68#10, 68#11, 69#3, 69#4, 69#6, 69#7, 69#21, 70#1, 70#7, 70#15, 70#34

Recording order is the frozen catalog order within each cell. Within each recording: segment 20 through 59, then local trial in ascending order for each segment; preserve recording-local IDs and cell-global trial offsets. Source ID: recording-live-frames-{150*segment:06d}-{150*segment+149:06d}-trial-{one_based_trial}. Expected 5480 sequences, 657600 valid response bins across the 137 recording-local trials. No cell/recording/trial exclusion.

## Production evaluation contract

Native 150 Hz; 150 bins/sequence; reset each 1 s sequence; score local 30..149 only. No cross-sequence pre-roll or state continuation. Original calibrated 17×17 central L+M Weber drive from the 51-pixel crop and 3-pixel pooling; original blank calibration, signs and coordinates. Same frozen parser, floor binning and Bernoulli event target q=1[count>0]. Full unscored warmup is supplied as input and history. Observed spike history is strictly past, implemented by the existing model; no current/future target leakage and no generated-spike replacement. All four models use identical events/masks/source/trial order. Construct the new split using existing loader semantics over the first 60 live seconds, selecting segments 20..59; this only extends data coverage, with no training invocation.

NLL is the existing CPU Torch 2.6.0 expected_bernoulli_nll: sum_valid(softplus(z)-q*z)/number_valid, nats per valid bin. Float32 model and original tensor reduction; no probability clipping or new loss. Within a cell weight all valid recording/trial bins as production; population means weight each of 22 cells equally. Cells share a stimulus, and cell-bootstrap intervals are not new-animal or new-movie-sample generalization intervals.

## Frozen models and preflight

Aligned checkpoint root was resolved from the fixed-alignment analysis-manifest.json inputs_sha256, not inferred from a directory name. Both Canonical families must have public name Canonical V1, revision 4, h1-shared-bc-direct-broad-ac and bc-central-disk_ac-overlapping-full-disk, 33 trainable scalars and no trainable center. Aligned positions equal the frozen LN final center mapping x_deg=x_LN*(3*4.6/256), y_deg=-y_LN*(3*4.6/256); no refit. All final fresh full-train checkpoints and their initial hashes are fixed below.

| Cell | Model | Frozen checkpoint |
|---|---|---|
| 67#4 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_4/model-trained.pt |
| 67#4 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/67_4/model-trained.pt |
| 67#4 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/67_4/ln-trained.pt |
| 67#4 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/67_4/cnn-trained.pt |
| 67#6 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_6/model-trained.pt |
| 67#6 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/67_6/model-trained.pt |
| 67#6 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/67_6/ln-trained.pt |
| 67#6 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/67_6/cnn-trained.pt |
| 67#7 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_7/model-trained.pt |
| 67#7 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/67_7/model-trained.pt |
| 67#7 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/67_7/ln-trained.pt |
| 67#7 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/67_7/cnn-trained.pt |
| 67#14 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_14/model-trained.pt |
| 67#14 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/67_14/model-trained.pt |
| 67#14 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/67_14/ln-trained.pt |
| 67#14 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/67_14/cnn-trained.pt |
| 67#21 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_21/model-trained.pt |
| 67#21 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/67_21/model-trained.pt |
| 67#21 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/67_21/ln-trained.pt |
| 67#21 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/67_21/cnn-trained.pt |
| 67#26 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_26/model-trained.pt |
| 67#26 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/67_26/model-trained.pt |
| 67#26 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/67_26/ln-trained.pt |
| 67#26 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/67_26/cnn-trained.pt |
| 67#33 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_33/model-trained.pt |
| 67#33 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/67_33/model-trained.pt |
| 67#33 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/67_33/ln-trained.pt |
| 67#33 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/67_33/cnn-trained.pt |
| 67#34 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_34/model-trained.pt |
| 67#34 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/67_34/model-trained.pt |
| 67#34 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/67_34/ln-trained.pt |
| 67#34 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/67_34/cnn-trained.pt |
| 68#3 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_3/model-trained.pt |
| 68#3 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/68_3/model-trained.pt |
| 68#3 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/68_3/ln-trained.pt |
| 68#3 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/68_3/cnn-trained.pt |
| 68#4 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_4/model-trained.pt |
| 68#4 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/68_4/model-trained.pt |
| 68#4 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/68_4/ln-trained.pt |
| 68#4 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/68_4/cnn-trained.pt |
| 68#7 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_7/model-trained.pt |
| 68#7 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/68_7/model-trained.pt |
| 68#7 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/68_7/ln-trained.pt |
| 68#7 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/68_7/cnn-trained.pt |
| 68#10 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_10/model-trained.pt |
| 68#10 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/68_10/model-trained.pt |
| 68#10 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/68_10/ln-trained.pt |
| 68#10 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/68_10/cnn-trained.pt |
| 68#11 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_11/model-trained.pt |
| 68#11 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/68_11/model-trained.pt |
| 68#11 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/68_11/ln-trained.pt |
| 68#11 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/68_11/cnn-trained.pt |
| 69#3 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_3/model-trained.pt |
| 69#3 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/69_3/model-trained.pt |
| 69#3 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/69_3/ln-trained.pt |
| 69#3 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/69_3/cnn-trained.pt |
| 69#4 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_4/model-trained.pt |
| 69#4 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/69_4/model-trained.pt |
| 69#4 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/69_4/ln-trained.pt |
| 69#4 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/69_4/cnn-trained.pt |
| 69#6 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_6/model-trained.pt |
| 69#6 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/69_6/model-trained.pt |
| 69#6 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/69_6/ln-trained.pt |
| 69#6 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/69_6/cnn-trained.pt |
| 69#7 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_7/model-trained.pt |
| 69#7 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/69_7/model-trained.pt |
| 69#7 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/69_7/ln-trained.pt |
| 69#7 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/69_7/cnn-trained.pt |
| 69#21 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_21/model-trained.pt |
| 69#21 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/69_21/model-trained.pt |
| 69#21 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/69_21/ln-trained.pt |
| 69#21 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/69_21/cnn-trained.pt |
| 70#1 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_1/model-trained.pt |
| 70#1 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/70_1/model-trained.pt |
| 70#1 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/70_1/ln-trained.pt |
| 70#1 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/70_1/cnn-trained.pt |
| 70#7 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_7/model-trained.pt |
| 70#7 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/70_7/model-trained.pt |
| 70#7 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/70_7/ln-trained.pt |
| 70#7 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/70_7/cnn-trained.pt |
| 70#15 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_15/model-trained.pt |
| 70#15 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/70_15/model-trained.pt |
| 70#15 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/70_15/ln-trained.pt |
| 70#15 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/70_15/cnn-trained.pt |
| 70#34 | zero | output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_34/model-trained.pt |
| 70#34 | aligned | output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/cells/70_34/model-trained.pt |
| 70#34 | LN | output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/70_34/ln-trained.pt |
| 70#34 | CNN | .omo/evidence/compact_causal_cnn_baseline/cells/70_34/cnn-trained.pt |

Before any held-out prediction/metric, replay original development validation [16,20) for all four families. Canonical zero/aligned: strict load, exact target/mask/source/trial order, bitwise logits equality and exact original NLL. LN: strict final-refit load and exact original NLL. CNN: strict final load, original GPU float32 forward (Torch 2.10.0+cu126, RTX4070 Laptop, deterministic algorithms, no AMP/TF32, CUBLAS_WORKSPACE_CONFIG=:4096:8), then original CPU Torch2.6.0 NLL on returned full-precision logits, compared exactly to existing results.json. Batch size 8 for all original evaluation functions and held-out conditions. Canonical/LN forward use original CPU Torch2.6.0 with 2 threads. Constructor RNG uses only existing checkpoint/training seed and is overwritten by strict frozen state; no new model seed. No checkpoint conversion, precision relaxation, backend-search rescue or altered pass tolerance. Any required mismatch or unavailable original runtime stops the entire experiment before held-out inference.

## Frozen structural interventions

Only aligned Canonical: normal (empty clamps); H1-off={PathwayClamp.H1}; direct-BC-off={DIRECT_BC_SUSTAINED,DIRECT_BC_TRANSIENT}; AC-off={AC_LOCAL,AC_TRANSIENT}. Use the production forward_sequence clamps argument. Direct BC branches are removed at direct pathway-current readout, preserving broad BC drive to AC. All bias/history parameters, remaining pathway gains and all state_dict tensors are unchanged; no refit or recalibration. No combined clamps beyond the two declared grouped branches. Every condition uses the same sequence/history/mask and batch order. Logit effect uses off-normal over the complete scored-bin vector (warmup excluded); store signed mean, mean absolute value, RMS and full vector L2 norm.

## Estimands and statistics

Primary paired differences per cell: aligned-zero, aligned-LN, aligned-CNN; negative means lower aligned NLL. Report all per-cell NLLs/differences, mean, median, negative/positive/exact-zero counts and paired-cell mean and median 95% percentile bootstrap intervals. Optional Constant is the explicitly requested analytical constant fit to identical scored test targets: p=their event fraction per cell; report its Bernoulli entropy as a test-fitted descriptive reference, not a frozen externally fitted predictor. No optimizer, checkpoint or tuning for Constant.

Bootstrap: numpy default_rng(2026090501), 100000 draws of 22 complete biological-cell pairs with replacement in the frozen cell order; use the same index matrix for all paired model and pathway differences. Percentiles 2.5 and 97.5, NumPy default linear quantile method. This seed is solely for requested statistics, not a model seed. No p-values and no group significance tests.

Secondary alignment: frozen radial_offset_degrees from the existing fixed-alignment audit per_cell_results.csv against I=zero-aligned; descriptive Pearson and Spearman (average ranks for ties). Single outlier diagnostic only: drop the cell with largest held-out absolute I (ties by lexical cell_id); recalculate mean I, Pearson and Spearman once. No other deletion search. Compare existing development aligned-zero mean difference, wins, Pearson and Spearman with the same four held-out quantities. Do not rerun RF/temporal/center estimation.

Pathways: Delta NLL=off-normal; positive means the frozen removal worsens predictive likelihood. Per pathway report equal-cell mean/median, positive/negative/exact-zero counts, the paired-cell mean and median percentile intervals and MC ON/MC OFF/PC ON/PC OFF descriptive means. Per cell/pathway retain signed mean Delta logit, mean absolute Delta logit, RMS, full valid-vector norm and Delta NLL. Descriptive Pearson/Spearman of mean-absolute-logit effect versus Delta NLL, separately for each pathway, without thresholds or p-values.

## Interpretation fixed before results

Prediction: REPLICATED requires lower held-out aligned mean NLL, the mean-difference interval below zero (the conservative route to the user's CI-or-strong-cell-support criterion), positive mean improvement after the single declared deletion, and a clearly positive offset relation supported by Pearson and Spearman including that diagnostic. PARTIAL_REPLICATION applies if mean improvement remains but intervals/cell support or offset relation weaken markedly or the single cell dominates. FAILED_TO_REPLICATE applies if improvement disappears/systematic worsening occurs or the offset relation disappears. Preserve the user's qualitative 'clearly positive' criterion: report all coefficients and make a bounded descriptive judgment, without inventing a new correlation significance threshold or changing statistics after results.

Pathways are judged independently: HELD-OUT PREDICTIVE CONSEQUENCE SUPPORTED when mean off-normal NLL is positive and its prespecified 95% interval lies above zero; WEAK / MIXED when positive effects are unresolved or heterogeneous; NO POSITIVE PREDICTIVE SUPPORT when no positive population degradation occurs or removal improves prediction. Always report sign heterogeneity. Clamp findings are predictive consequences of model-internal intervention under frozen remaining parameters/bias and depend on this model family; no biological necessity, unique causal contribution, retrained ablation or real-retina lesion claim. Baseline intervals containing zero are unresolved/not separated by the current interval; no equivalence or superiority claim.

## Consumption and stop rules

Immediately upon the first successful held-out target-based NLL, create TEST_CONSUMED.md before viewing/reporting its value: exact range/identities/models/clamps/metrics, branch/HEAD, this SHA256 and UTC time. If later execution fails, preserve partial consumption and stop; do not silently restart or select a new range. The full declared range is conservatively retired after first consumption. No consumption marker if a preflight gate fails before any held-out output. Final REPORT answers only the seven requested questions plus one research-decision sentence. New training, tuning, checkpoint selection, parameter/production edits, RF/tau/delay audits, new synthetic/illusion runs and additional experiments remain zero.
