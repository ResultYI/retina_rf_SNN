# Phase M1 — fixed BC versus downstream AC context selectivity

## Authorization and provenance

Only original Phase M1 attachmentd40dd7d5-9f7c-49d5-b42a-9c7ccbe8fbc7 plus the user's explicit deterministic-derivation continuation are authorized. The preceding run stopped before a full PROTOCOL.md existed; this document first records the original M1 definitions before any selectivity analysis. Preserve the stopped preflight/report/manifest as historical copies. No original metric is replaced.

Historical evidence: aligned final checkpoints resolved from the prior held-out evidence manifest, original development causal-replay-tensors.pt and validation-predictions.pt, frozen training-only bias_fit_per_cell.csv and bias_fit_lock.json. New evidence: development recalibrated logits/NLL and M1 statistics. Label derived outputs NEW_DETERMINISTIC_DERIVED_ARTIFACT, never historical replay or independently frozen historical recalibrated predictions.

Training, bias fitting, optimizer execution, model/architecture/front-end/center/support/radius/checkpoint/production changes, new seeds, new RF/tau/delay analyses, pathway retraining, illusion/synthetic/artificial stimuli and M2 remain0. No H1 primary analysis, no bias selection using development targets. Normal is uncalibrated frozen aligned normal, consistent with the previous bias-control primary reference.

## Preconditions and deterministic derivation

Record branch/HEAD/dirty status and hashes of original sources/data/checkpoints/raw tensors/targets/masks/bias artifacts. Verify all44 biases (22direct-BC,22AC), exact cell/condition identity, CSV hash against bias_fit_lock and prior manifest. Parse original full-precision CSV floats, never report-rounded numbers. No solver call.

Rebuild development with original production loader; verify input/target/mask/source/trial order. Strict-load aligned checkpoint, same frozen center/support/parameters. In original CPU Torch2.6.0+cpu/2threads, replay original normal batch8 evaluation against validation-predictions and original NLL; replay direct_BC_off/AC_off as full validation batches, matching their archived producer call and all logits bitwise. Require original causal normal also equals original validation normal. All remaining parameters/state unchanged; no gradients. Any mismatch STOP_UNVERIFIED, no backend/tolerance search.

Construct only z_recal=z_off.double()+frozen_training_bias, using float64 as in prior bias-control statistical calculation. Save full-shape raw normal/selected raw off/recalibrated logits with targets, mask, source IDs and trials. Arithmetic check must match the independently reconstructed float64 addition bitwise. Subtraction (z_recal-z_off)-bias may have IEEE754 rounding; require its maximum absolute error <=4*float64_eps*max(1,maxabs(z_off),abs(bias)), save actual error/bound, never change values to force exact subtraction. NLL=mean(softplus(z)-y*z) over original scored mask, float64 Torch2.6.0+cpu, identical to prior bias control. No NLL informs bias or analysis choices. Save derived logits, NLL CSV, source/target/bias provenance manifest and SHA256 lock before features/selectivity. Hash-check original inputs again. Update preflight to VERIFIED DERIVATION INPUTS — development recalibrated logits newly derived deterministically from frozen training-only biases.

## Data and fixed features

Primary development live[16,20)s, original sequence-local t30..149 scored mask, no T1 truncation:22cells/37recordings/137trials,548sequences,65760bins. Preserve original150Hz,150-bin resets and strictly-past observed event history. Original calibrated17x17 L+M Weber drive before model transforms. No physical photon-flux interpretation; frozen decoded live origin751 remains acquisition-provenance UNKNOWN.

Use exact checkpoint fixed binary BC/AC buffers; assert BC subset AC, nonempty BC and AC\\BC, finite coordinates/drive for all22; exact construction geometry enforced by strict-load. Any invalid support STOP, no alternative radius/annulus. Record centers/support sizes/tensor hashes and exact uniform weights.

C=uniform mean I over BC; R=uniform mean I over AC\\BC; primary S=abs(R-C); signed secondary Q=R-C; U=abs(C); V=abs(R). LSC_AC=sqrt(uniform mean over FULL AC of (I-mean_AC(I))^2). Float64, dimensionless Weber-drive units. S is the only primary feature. No learned weights, RF/Jacobian, pathway states, signed/ratio/product/variance primary replacement.

Per cell derive S,U,LSC_AC quintile edges only from respective development scored stimulus values, NumPy linear quantiles0.2/0.4/0.6/0.8; assign1+searchsorted(edges,value,side=right). Record actual tie counts; require increasing edges/nonempty development strata. Controls cannot change S thresholds. Save/hash-lock stimulus-only feature files and thresholds before accessing per-bin pathway loss. Save C,R,S,Q,U,V,LSC_AC,cell/group/frame/time/source/trial/sequence identity; events attached only afterwards.

## Predictive consequence and normalization

Float64 per-bin ell=Torch softplus(z)-y*z. Delta_P=ell_off_recal-ell_normal, P=direct-BC or AC. Positive means frozen model-internal pathway removal worsens prediction; not biological causal contribution. Same scored rows for all conditions. S-quintile tables give mean/median Delta,normal/recal-off NLL,event rate,C,U,R,V,LSC_AC,bin count. U-quintile table uses same quantities for secondary central-drive comparison.

E_P=mean(Delta_P|S-Q5)-mean(Delta_P|S-Q1). Denominator B_P=mean(Delta_P over ALL analyzed bins). N_P=E_P/B_P if B_P>0 and numerically resolvable; no epsilon/clipping. Operational numerical-zero bound is64*float64_eps*max(1,mean(abs(Delta_P))); a denominator <=bound is undefined. Save B,bound,defined status and raw E; do not select cells on denominator. D=N_AC-N_BC. Equal-cell population normalized summaries/CI require all22 defined; otherwise mark unavailable with counts, never silently use a selected subset. Raw E remains22-cell.

## Matched controls and secondary analyses

U primary control: inside each of5development U strata compare SAME global S-Q5/Q1. Only strata with both groups contribute; a=min(nQ1,nQ5),weight=a/sum(a). E_Umatched is weighted within-stratum high-minus-low Delta. Use the SAME unconditional B_P denominator for N_Umatched, then D_Umatched=N_AC_Umatched-N_BC_Umatched. Save overlap mass/fractions/counts/weights and residual U difference (plus C/R/LSC descriptors). If any development cell lacks U overlap STOP_UNVERIFIED instead of dropping it. Coarse strata do not equalize continuous magnitude exactly.

LSC secondary control: same global S groups within5fixed LSC_AC strata with min-count weights. Save E_LSCmatched,overlap,residual LSC. No bootstrap/outlier for this descriptive control; absent overlap means unavailable without selected-subset population mean. It cannot rescue or independently establish GO. Disappearance of the AC effect is an explicit interpretation downgrade and prevents the full six-condition AC-context claim; do not silently ignore it.

Secondary G_P=mean(Delta_P|U-Q5)-mean(Delta_P|U-Q1), descriptive mean/median/signs only. Per-cell Spearman(S,Delta_AC/BC),Spearman(U,Delta_AC/BC), average tied ranks; population mean/median/sign counts, no bin-wise significance. Signed context: Q<0 versus Q>0 AC mean/median predictive penalty and counts; exclude exact-zero Q from these two groups but record zero count. No further signed bins and no signed contribution to verdict.

MC_ON/MC_OFF/PC_ON/PC_OFF: descriptive raw E_AC/E_BC,N_AC/N_BC,D mean/median/N, no group tests/cohort changes. Only two outlier diagnostics: delete one maximum-absolute E_AC cell and one maximum-absolute D cell (lexical tie-break), remean once; D diagnostic unavailable if D lacks complete22. No iterative, BC,U,LSC,G or consumed deletions.

## Paired-cell uncertainty and verdict

Reuse existing bootstrap seed2026090501 and100000x22 index matrix SHA256 e7c2af0a9e6c1c54358f3f64513ad92790ea4bcc7647e92f2c9ab3a9da901528. Equal biological-cell weights with paired values retained, linear2.5/97.5 percentiles, no new seed/p-values. Mean/median/sign counts for E_BC,E_AC,N_BC,N_AC,D and U-matched counterparts; mean/median95% bootstrap CI for raw E_BC/E_AC/D and U-matched E_BC/E_AC/D. N is displayed without extra CI. No new bootstrap on descriptive LSC/G/signed/class/consumed results.

Before results, operationalize the original qualitative evidence requirements consistently: positive CI means mean95% lower endpoint>0; substantial U retention means positive U-matched mean CI and at least50% of raw AC mean; clear D direction means D mean95% CI>0 and mean after its single deletion>0. Full GO — AC CONTEXT SELECTIVITY SUPPORTED requires raw E_AC meanCI>0, its deletion mean>0, substantial U retention, clear D direction, and LSC-matched AC mean>0 (available all22). Raw E and U/D evidence are primary; LSC can only limit interpretation, not supply missing primary evidence. Report which of six original conditions pass individually.

If full GO fails but raw AC mean E>0, use MIXED — FUNCTIONAL SELECTIVITY WEAK OR NONSPECIFIC (including positive but BC-shared, U-attenuated, LSC-explained or unresolved differentiation). Otherwise NO-GO — NO DISTINCT AC CONTEXT SELECTIVITY. This ordered convention resolves the original overlapping illustrative MIXED/NO-GO cases before observing M1 results. No subset/metric/support/bias/threshold revisions to alter verdict. Undefined normalized values preclude GO, not raw-statistic reporting.

## Development lock, consumed reuse and deliverables

Finish ALL declared development analyses and write development_verdict_lock.json containing verdict,all primary numbers,protocol hash and development evidence hashes; hash that file separately. Only then read consumed[20,60) pathway prediction payloads. Consumed uses SAME geometry/features/trainingbiases and frozen numerical development S/U/LSC thresholds, original120scored bins/sequence,657600bins. Use saved prior held-out raw logits; deterministic float64 bias addition and NLL must match prior bias-control summaries exactly. Do not fit or run new held-out inference. Denominators use each split's ALL analyzed bins under the same formula. Report E_AC/E_BC/D population-direction and per-cell sign consistency, descriptive only, no fresh confirmation or altered verdict.

Write requested stimulus_features.parquet and tables, derivation provenance/locks, updated preflight, REPORT/manifest in the requested directory. Preserve old stop records in a historical subdirectory before replacing explicitly updated status/report. REPORT answers exactly10 M1 questions, then whether evidence supports M2. No artificial stimuli, model change, training or automatic M2.
