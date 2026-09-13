# Phase R1b protocol: MODEL_NATIVE_REPEATABILITY_CEILING

Artifact class: NEW_DETERMINISTIC_DERIVED_ARTIFACT. Frozen before any R1b reliability or model score. This protocol implements the user's Phase R1b request, with numerical conventions below fixed in advance.

This analysis does not reproduce the publication's original retinal-reliability estimator. It defines a separate model-native repeatability benchmark at the exact temporal and target resolution used by the current prediction task.

## Scope and gates

R1 remains untouched and UNVERIFIED / UNRESOLVED. No training, fitting, new optimization seed, architecture/checkpoint/center/front-end/history modification, pathway analysis, synthetic data or new test data. All files written only to this new audit directory and its dedicated planning files. Official bounded scan must return OFFICIAL_IMPLEMENTATION_NOT_RECOVERED; discovering an exact raw-repeat reliability routine stops R1b.

Use exactly R1's 20 eligible biological cells, 120 live repeats: MC ON 4, MC OFF 4, PC ON 9, PC OFF 3. Exclusions 67#4 and 68#10: LOCAL_PUBLIC_RAW_REPEAT_FILE_UNAVAILABLE. Reverify all R1 input hashes and artifact identities. No inclusion changes. Current repeated recordings are consumed biological data, not an untouched predictive test.

## Exact computational target and time contract

Reuse data.schottdorf_lee_spikes.parse_recording_spike_trials and data.schottdorf_lee_multirecording._bin_trial/load_schottdorf_cell. Raw timestamps are integer ticks times 0.1 ms in float64, already relative to each movie repeat; Video Starts is recorded metadata, not an additional offset. Only six live columns with 0 <= time_ms < 60000; seventh maintained-activity column excluded.

Production index is floor(time_ms * 150.0 / 1000.0), retaining 0 <= index < 9000. Nominal bin is [t*1000/150,(t+1)*1000/150) ms, with production floating operation order authoritative at boundaries. Do not replace with rounded edges or epsilon shifts. Occupancy = (count > 0).float32. Save exact counts and occupancy for all 9000 bins and all repeats. Boundary evidence includes actual timestamps on boundaries and adjacent representable values; no synthetic response experiment.

dt=1000/150 ms; production decoded frame zero is 751, unchanged. FRAME_ZERO_ACQUISITION_PROVENANCE = UNKNOWN. Exact existing calibrated L+M Weber movie preprocessing, crop 51/pool 3, 17x17 cones. No frame offset comparison or interpolation.

Segment live [0,60) into 60 independent 150-bin sequences; reset all state at each sequence start. Score bins 30..149 within every sequence. Per repeat: 9000 total, 7200 scored; per cell: 43200 scored observations; cohort: 864000. Apply the same score mask to every biological statistic and model comparison.

Before correlations or inference, compare raw-reconstructed targets/counts/masks/source identities against the production adapter for [0,16), [16,20), [20,60). Compare every available saved validation and consumed [20,60) target tensor by recording/local-repeat/source ID. Training targets are recreated by the unchanged production adapter because the training runner does not persist training target tensors; distinguish live adapter equality from saved-tensor equality. Any mismatch stops all downstream scoring. Validate every repeat's stimulus trace exact identity and reference movie identity.

## Repeatability and numerical conventions

Primary occupancy target SIX_REPEAT_MEAN_OCCUPANCY: ybar(t)=mean_r y_occ(r,t), a finite-repeat empirical estimate of stimulus-locked event probability, not true ground truth firing probability. No smoothing.

Enumerate all ten complementary 3-vs-3 partitions once: lexicographically ordered A triples containing repeat 1, B=complement. Pearson over scored bins computed in float64. Save all ten split values. r3=tanh(mean(atanh(r_s))); raw-r arithmetic mean diagnostic only. No clipping at any stage. A constant trace or nonfinite correlation makes normalization undefined. If any |r_s|>=1, retain raw values and flag the Fisher estimator undefined rather than introducing epsilon clipping.

Rel6=2*r3/(1+r3). Denominator magnitude <= float64 epsilon or nonfinite -> undefined. Rel6<=0 -> normalized correlation undefined. C=sqrt(Rel6) for finite positive Rel6; Rel6>1 is retained with FINITE_REPEAT_ESTIMATOR_ANOMALY, and its algebraic normalization is reported as anomalous. Do not substitute 1.

Apply identical split-half procedure separately to counts and extra counts e=max(n-1,0). Their variance scales are not compared directly. All components use exactly the scored bins. Multi-spike-bin fraction=mean(n>1) over 6*7200 scored bins. Also report conditional fraction among occupied bins. Beyond-first spike fraction=sum(e)/sum(n). Event rate=150*mean(occupancy); firing rate=150*mean(count). Undefined denominator -> missing with explicit status.

## Frozen model contracts and metrics

Resolve 22 aligned checkpoints through the fixed-alignment analysis/artifact manifests and match R1 checkpoint hashes; LN/CNN through the prior heldout evidence manifest. Evaluate only the eligible 20. Use each model's native frozen runtime: Canonical and LN CPU torch 2.6.0, CNN original deterministic CUDA torch 2.10.0+cu126, TF32 off, cuDNN benchmark off. Strict-load full states, eval, no gradients, no optimizer, verify states unchanged. Existing constructor seeds only, overwritten by frozen state; no new optimization seed. Batch size 8 as existing evaluation.

First run STIMULUS_ONLY_ZERO_HISTORY: force observed occupancy history inputs to zero for all three models, exact sequence reset and masks. Same repeated movie yields one common deterministic trace. Save logits and sigmoid probabilities without output recalibration. Freeze all primary prediction and score files before running secondary FORMAL_CONDITIONAL_HISTORY.

Raw Pearson r_model=Pearson(p,ybar), secondary Spearman, mean p and mean occupancy. NCorr=r_model/C only when defined, retaining negative and >1 values. NCorr is an attenuation-style normalization relative to the estimated reliability of the finite six-repeat mean, not a biological percentage uniquely determined by the retina. Its classical interpretation assumes independent additive repeat noise; same-dataset fitting and heteroscedastic Bernoulli noise limit that interpretation.

Second primary-support statistic: s_t^2=sum_r(y_rt-ybar_t)^2/(R-1), noise=mean_t(s_t^2)/R, Vsignal=Var_t(ybar,ddof=0)-noise; MSEsignal=mean_t((ybar-p)^2)-noise; FEV=1-MSEsignal/Vsignal only if finite Vsignal>0. No clipping. This is a project-specific model-native noise-corrected statistic, not the publication's metric. Preserve FEV<0 or >1. Save all intermediate terms. Report Pearson/NCorr and FEV together, never choose retrospectively. A substantive navigation conflict -> UNRESOLVED_METRIC_DEPENDENT.

Secondary conditional: each repeat's strictly-past observed occupancy, same forward/reset. Report per-repeat Pearson to corresponding occupancy and to six-repeat mean. Report conditional-minus-zero association using matched response targets; no primary ceiling normalization. Observed history is response-derived information, not additional stimulus-only explained fraction.

## Summaries and uncertainty

Biological cell is the population unit. Equal-cell means and medians for raw Pearson, NCorr, FEV; group descriptive summaries for four fixed classes. Pairwise Canonical-LN, Canonical-CNN, CNN-LN mean differences on common finite cells for each metric. Paired-cell percentile bootstrap 100000 draws, NumPy default_rng seed 20260907, 2.5/97.5 percentiles with linear quantile method. Generate one index matrix per common cell count; no optimization/selection seed. Report valid n and missing cells.

Only leave-one-repeat-out sensitivity: six five-repeat means, model raw Pearson and min/max/range/std(ddof=0); no five-repeat reliability normalization. No time-bin bootstrap, repeated-split pseudo-replicate CI, p-values, thresholds search, or unrequested stratification.

For each model and gap (1-NCorr,1-FEV), descriptive Pearson/Spearman across cells against multi-spike-bin fraction, beyond-first fraction, extra-count Rel6 and event rate; pairwise finite filtering with n. Save extra-count raw r3 even if Rel6 invalid.

## Navigation and reporting

Use the user's qualitative verdict definitions without inventing numerical decision thresholds: NEAR MODEL-NATIVE OCCUPANCY CEILING; SHARED LARGE OCCUPANCY GAP; CANONICAL-SPECIFIC OCCUPANCY GAP; OBSERVATION-TASK LIMITATION SIGNAL (separate observation verdict only if occupancy is near its ceiling); UNRESOLVED_METRIC_DEPENDENT; UNRESOLVED. Two primary metrics must support a strong directional claim. Paired bootstrap quantifies model differences, not full ceiling-estimator uncertainty. Positive repeatable extra counts alone do not prove the observation contract is the main prediction limitation or that Poisson is correct.

REPORT contains the mandatory opening disclaimer, exactly twelve numbered question answers, then CORE OPTIMIZATION DECISION and OBSERVATION MODEL DECISION. No automatic next experiment. Final evidence manifest hashes raw data, movie, checkpoints, production source, protocol, R1 artifacts and all outputs; retain raw arrays, model traces, split values and runtime evidence for independent recomputation.

## Execution clarifications before freeze

Pass unshifted same-repeat occupancy to native observed-history APIs; each native API implements its own strictly-past shift. Primary uses explicit zeros, never the ordinary evaluation helper that supplies observed events. Join saved tensors by recording ID, live-frame interval, and one-based local repeat, not global concatenated trial indices. Only the selected repeated recording enters the scoring cohort.

Unsupported required runtime, absent required input, nonfinite predictions or failed integrity checks stop affected computation and produce UNRESOLVED; no runtime substitution selected from scores. If qualitative navigation categories cannot be supported clearly by both prespecified metrics, report UNRESOLVED rather than inventing a post-result threshold. All original input and checkpoint hashes and R1 files are rechecked after completion.
