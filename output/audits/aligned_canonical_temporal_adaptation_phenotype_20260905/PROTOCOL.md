# Phase T1 — temporal/adaptation-dependent residual phenotype

## Scope and frozen identities

Execute only Phase T1, as authorized by attachment5eccde2b-efa4-4942-b79e-2fba65cf75f3. Training, parameter fitting, architecture/front-end/core/checkpoint/production changes, new seeds, trainable centers, illusion/synthetic probes, new RF/tau/delay audits and pathway clamp analyses remain0. No Phase T2 implementation even under GO. Zero-center is not evaluated. No calibration biases from the prior bias-control task.

Resolve all22 aligned Canonical V1 final checkpoint paths, and final fresh full-train LN/CNN paths, from the preceding held-out evidence_manifest.json checkpoint_identity records, whose aligned lineage is grounded in macaque_fixed_alignment_experiment_20260905. Record branch, HEAD, dirty status, model/data/source/checkpoint SHA256, aligned centers, support geometry and exact weights before execution. Preserve all old artifacts. Canonical revision4, h1-shared-bc-direct-broad-ac, bc-central-disk_ac-overlapping-full-disk,33 learned scalars with fixed LN-derived centers.

## Exact replay and data

Before temporal phenotype features/losses, rebuild development through the original production loader and strict-load all66 final cell/model checkpoints. Require exact input, target, original valid mask, recording/source/trial order, bitwise original logits and original NLL. CPU Torch2.6.0+cpu/2threads/batch8 for aligned/LN; original deterministic GPU Torch2.10.0+cu126/RTX4070 Laptop/batch8 for CNN, then original CPU NLL reporting. Seeds used by constructors are the existing frozen seeds; no new seed. Assert state unchanged and no gradients. Any mismatch: STOP; no tolerance relaxation, alternate backend search or phenotype analysis.

Primary is development live[16,20)s,frames[2400,3000),decoded[3151,3751),22cells/37recordings/137recording-local trials/548sequences. Each sequence has150bins at150Hz; original valid mask scores local t30..149,65760 original scored bins. Preserve the original calibrated17x17 L+M Weber drive, trial multiplicities and strictly-past observed-event model history. No state/history/reset changes to inference. Frozen decoded live origin751 remains acquisition-provenance UNKNOWN.

The phenotype mask for BOTH A and C is original_valid_mask AND sequence-local t>=45. Every included bin has45 real sequence-local past bins. Never concatenate sequences; no padding or fabricated preceding state. Thus local t45..149 are included:105bins/sequence,57540 development phenotype bins. Save full Boolean phenotype masks and original/retained/excluded counts per cell/split; original NLL exact replay still uses its original120bins/sequence.

## Only two temporal features, stimulus-only

I[t,x] is original dimensionless L+M Weber drive before any model transform. For each aligned cell, w=checkpoint model['feature_bank.bc_support'][0]/support.sum(), a fixed binary BC central disk (source radius0.06degrees midget/PC;0.10degrees parasol/MC). All weights nonnegative, sum1; record center, coordinate/support hashes, support size and exact weights. No AC, learned RF/Jacobian or temporal/pathway-state weighting.

Compute float64 local M[t]=sum_x w[x]I[t,x]. U[t]=abs(M[t]) is the current absolute-drive control. These are not physical photon flux estimates.

Feature A, PRIMARY adaptation: A[t]=sqrt(mean_{j=1..45}(M[t-j]^2)). Strictly past; excludes current t;45bins=300ms. Unavailable local t<45 are masked out, not zero-padded. A is a stimulus-derived adaptation-load proxy, not a cone adaptation state or learned time constant.

Feature C, SECONDARY recent change: C[t]=abs(M[t]-M[t-5]),5bins=33.333ms. Same phenotype mask as A. No other windows/delays/temporal statistics as competing estimands; no scan or feature replacement. M,U,C,A and all thresholds are calculated without spikes, predictions, residuals or model/pathway state.

## Quintiles and feature lock

For each cell, independently derive A quintile edges from A alone, C edges from C alone, and U matching edges from U alone, using all development phenotype-mask rows including frozen trial multiplicities. NumPy linear quantiles at0.2,0.4,0.6,0.8; assign1+searchsorted(edges,value,side='right'). Ties go upward; report actual bin counts. Controls cannot define or change A thresholds. Require strictly increasing edges and nonempty five development strata for A/C/U. Stop if definitions fail; no alternative tie rules, cutoffs or cell exclusions.

Save/hash-lock stimulus-only feature files, exact weights, phenotype masks and numerical thresholds BEFORE reading per-bin losses. Spikes/events are attached only afterwards. Quantile tables for A and C contain all3 NLLs,2relative losses,event rate,M,U,C,feature mean and bin counts. Retain frame/time/cell/group/recording/source/local/global trial/sequence identity in compressed outputs.

## Loss and statistics

On the same phenotype rows, ell=logaddexp(0,z)-y*z in float64, using exact saved/replayed float32 logits and binary event y. D_LN=ell_aligned-ell_LN; D_CNN=ell_aligned-ell_CNN. Positive means Canonical has higher loss. Report the numerical difference between float64 loss aggregation and native float32 aggregation separately; it cannot relax exact replay.

Primary E_A=mean(D|A-Q5)-mean(D|A-Q1), for each comparator/cell. Secondary E_C uses C-Q5/C-Q1. Population equally weights all22 biological cells, not bins/trials. For primary raw A and U-matched A report mean,median,positive/negative/zero cells,100000 paired-cell percentile bootstrap mean and median95% intervals. Reuse existing seed2026090501, NumPy default_rng and the exact100000x22 index convention from prior tasks; SHA256 e7c2af0a9e6c1c54358f3f64513ad92790ea4bcc7647e92f2c9ab3a9da901528. Preserve within-cell pairing across both comparators/raw/matched. Linear2.5/97.5 percentiles. No new seed or p-values. Secondary C has descriptive mean,median and signs; no additional bootstrap is needed for it.

## Current-drive matched PRIMARY control

Use fixed development U quintile strata. Inside each, compare the same GLOBAL A-Q5 against A-Q1. A stratum contributes only with both groups present. a_s=min(n_AQ1,s,n_AQ5,s),alpha_s=a_s/sum(a_s). E_A_matched=sum(alpha_s*(mean(D|AQ5,s)-mean(D|AQ1,s))). No random sampling/outcome matching/fitting. Save all counts,weights,retained strata,overlap mass and mass/Q1 and mass/Q5 fractions. Save raw and weighted remaining U,M,C differences between high/low groups. Quintile matching is coarse; it cannot claim equal continuous current drive or equal signed M.

If any development cell lacks ANY overlapping U stratum, STOP with control/definition UNVERIFIED; do not drop the cell or invent matched value. Both comparators use identical stimulus-only weights. Primary matched bootstrap uses all22 cells. No matching of event rate and no rate model.

## Secondary temporal-change control and trends

Operationalize the requested 'large C difference' before outcomes: for each cell compute abs(mean(C|A-Q5)-mean(C|A-Q1))/population_sd(C over phenotype rows), using ddof0. If this reaches0.5 in at least one development cell, run ONE fixed U-quintile x C-quintile descriptive stratification for ALL22 cells, again comparing global A-Q5/Q1 and weighting each of25 strata by min(nQ1,nQ5). This trigger uses only stimuli and does not select cells. If none reaches0.5, record not-triggered and do not run2D contrasts. Zero sd with zero difference means0; otherwise invalid definition stops. Freeze the trigger with the feature lock. No bootstrap/outlier/significance/GO contribution from2D matching. Save overlap and residual U/M/C differences; mark unavailable if a cell has no overlap and do not report an all-cell population2D mean over a selected subset. Once triggered, the same descriptive procedure applies to consumed data regardless of its trigger statistic.

Compute per-cell Spearman(A,D_LN),Spearman(A,D_CNN),Spearman(C,D_LN),Spearman(C,D_CNN), with average ranks for ties. Describe mean/median/sign counts across cells only; no bin-wise significance. MC_ON/MC_OFF/PC_ON/PC_OFF summaries only for primary A raw and U-matched E: mean,median,N; no tests or selected cohort. If C-only effects occur, describe fast-change-related effects and do not call them adaptation.

Only outlier diagnostic: for each of the TWO raw primary adaptation E vectors remove the single maximum-|E| cell (lexical cell_id tie-break) and recompute mean once. No matched/C/2D/consumed deletion diagnostics and no repeated deletion.

## Development-only deterministic verdict

GO — TEMPORAL/ADAPTATION PHENOTYPE SUPPORTED requires at least one comparator with raw A mean95% CI strictly >0, single-deletion mean>0, U-matched A mean95% CI strictly>0, and matched mean>=50% of raw mean. The prespecified50% rule operationalizes substantial retention as in the preceding phenotype protocol. Relative loss, not increased absolute Canonical loss, is required. Disclose disagreement between comparators and residual confounding. This is support for a history-associated relative residual phenotype, not proof of missing photoreceptor adaptation.

If GO fails, MIXED — WEAK TEMPORAL PHENOTYPE applies when at least one comparator has positive raw A mean and either positive matched A mean or a raw mean95% CI strictly>0. This includes unresolved comparator-specific positive estimates or a raw supported effect attenuated/reversed by matching. Otherwise NO-GO — NO TEMPORAL/ADAPTATION-SPECIFIC DEFICIT and NO-GO FOR TEMPORAL/ADAPTATION FRONT-END. These ordered rules resolve the request's overlapping illustrative MIXED/NO-GO descriptions before results. C and2D controls remain secondary and cannot rescue adaptation GO or change this verdict. No label adjustment using cell classes or consumed data.

## Consumed reuse after locked development verdict

Complete ALL development primary/matched/secondary/trend/group/outlier analyses, then write development verdict and development_verdict_lock.json containing its SHA256 plus table/feature hashes. Only afterwards read consumed[20,60) phenotype data or predictions. Hashing old files for initial identity is not phenotype analysis.

Secondary consumed reuse uses the SAME centers/support/k5/window45/local mask and FROZEN numerical development A,C,U thresholds. Do not re-quantile consumed data. Same105bins/sequence,5480sequences,575400 phenotype bins (original657600). Rebuild input/target/mask/source/trial identity against prior saved consumed artifacts; reuse frozen normal aligned/LN/CNN logits with no new clamp or bias analysis. Report development and consumed descriptive means, direction consistency and per-cell sign consistency for A raw/U-matched and C raw;2D only descriptive if triggered in development. No consumed bootstrap/outlier/decision. If secondary groups or matching lack support, record unavailable; no selected-subset population mean. Consumed results cannot change frozen verdict or protocol and are not independent confirmation.

## Deliverables and stop

New directory aligned_canonical_temporal_adaptation_phenotype_20260905 only, plus scoped plan artifacts. Include PROTOCOL/hash,preflight,phenotype_mask_summary,compressed per_bin_temporal_features and schema,adaptation_quantile_nll,temporal_change_quantile_nll,per_cell_adaptation_excess_loss,population_adaptation_excess_loss,current_drive_matched_control,continuous_trends,cell_group_summary,development_verdict_lock,development_vs_consumed_descriptive,REPORT and evidence_manifest. Additional detail tables only implement declared controls/statistics. REPORT answers exactly9 requested questions, then only whether evidence is sufficient for Phase T2. No front-end implementation, training, automatic continuation or new model-form decision.
