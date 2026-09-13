# Bias-recalibrated held-out pathway clamp control

This is a predefined follow-up analysis on an already consumed confirmatory set, not an untouched test or a higher evidence tier. Only original-training-derived scalar additive logit biases may be fitted. There is no model training, optimizer, gain/slope/history/center/temporal/pathway refit, new model seed, architecture or production change.

## Frozen data and model

Use exactly the preceding macaque_aligned_heldout_pathway_evaluation_20260905 evidence_manifest.json, TEST_PROTOCOL.md, model_artifact_inventory.csv, test_source_ranges.csv and cells/*-test-predictions.pt. Their hashes and the preceding frozen input hashes are copied to this task's initial manifest before inference/fitting. The aligned checkpoint root is resolved from that manifest's 22 aligned checkpoint_identity records. Use the same final fresh full-train refits and LN-derived fixed positions; no trainable center. The prior protocol SHA256 is 3652c48aa4293f2c3503e15099b40657f6d3b396f4a06f4434c4f0e27167bb08.

All 22 cells, 37 recordings, 137 recording-local trials and exact order are those of the preceding protocol and source-range tables, incorporated by their frozen hashes. Original training: live [0,16) s, frames [0,2400), segments0..15. No inner-dev, original validation or heldout target enters scalar fitting. Test: consumed live [20,60) s, frames [3000,9000), decoded [3751,9751), segments20..59; 5480 sequences and 657600 scored bins. Bounds are zero-based half-open. Live origin751 retains acquisition-provenance UNKNOWN.

Preserve native150Hz, 150-bin sequence, reset at every sequence, first30 bins unscored, score30..149, calibrated17x17 L+M Weber front-end, original count binning and binary event target, strictly-past observed event history, recording/local-trial/global-trial/source order. Same CPU Torch2.6.0+cpu, float32 model forward, two threads, batch8. Existing constructor seed is only the original checkpoint's existing seed, overwritten by strict load; no new model seed or fit.

Conditions: aligned normal; H1-off={PathwayClamp.H1}; direct-BC-off={DIRECT_BC_SUSTAINED,DIRECT_BC_TRANSIENT}; AC-off={AMACRINE_LOCAL,AMACRINE_TRANSIENT}. Use the existing production forward_sequence clamps argument. All model state including output bias and history remains unchanged; fitted b is added externally to frozen logits.

## Mandatory replay gate before any fitting

Rebuild original training using the production loader and compare cone drive, counts, events, mask, source IDs and trial order exactly with previously verified original-training bundles. Confirm original aligned training contract/refit/checkpoint identities. Generate frozen normal/off training logits with the same history input convention.

Rebuild the exact heldout data from the production loader and compare target/mask/order with prior saved artifacts. Replay all 22 normal and 66 off logit tensors bitwise; recompute prior float32 NLL exactly. Verify 657600 scored bins. Any required mismatch stops the analysis, without different definitions, checkpoint conversion or tolerance relaxation. No bias fitting until every cell/condition passes.

Save training-only tensors in a distinct file. The fitting phase reads only this file plus the protocol and completed gate, never test targets or test logits. Hash-lock the completed scalar-fit file and fit table before any recalibrated heldout scoring. Test source IDs and tensor hashes are separately retained as provenance.

## Scalar solver fixed in advance

For each cell and each of four conditions (including normal diagnostic), solve mean(sigmoid(z_train+b))=mean(y_train) over the original valid mask. Treat frozen float32 logits as exact inputs and use float64 arithmetic for scalar fitting, rates and the control's Bernoulli NLL softplus(z)-y*z. This is the same likelihood; float64 avoids mixing precision across raw and shifted terms. Replay checks separately reproduce the preceding float32 NLL exactly, and the maximum raw float64-versus-original-float32 difference is recorded. No change to production model precision, implementation or checkpoints.

For observed rate p strictly between0 and1, use the deterministic guaranteed bracket [logit(p)-max(z), logit(p)-min(z)]. Bisection uses rate residual tolerance1e-12, bracket width tolerance1e-12 and at most200 iterations, identically for all cells/conditions. Stop successfully when absolute mean-rate residual<=1e-12; do not accept bracket width alone if the residual fails. Report total-event residual as valid_count times mean-rate residual. Degenerate all-zero/all-one targets have no finite optimum: record and stop rather than clip, cap or select a bias. No validation-driven stopping or pathway-specific tuning.

Save fitted scalar, residual, iterations, observed rate, pre/post predicted rate, train NLL before/after and actual bracket. Verify finite quantities and no increase in the convex training objective beyond1e-12. Normal uses the identical solver and data as a diagnostic, with no assumption its bias must be exactly zero. Normal recalibration is never the primary reference: every pathway contrast uses the original frozen normal logits.

## Fixed estimands and statistics

For each pathway: raw Delta=NLL(off)-NLL(original normal); recal Delta=NLL(off+b_train)-NLL(original normal). R=1-recal_Delta/raw_Delta only for raw_Delta>0; otherwise undefined/blank. Never clip R, and never call it causal variance explained. Population absolute reduction=mean(raw_Delta)-mean(recal_Delta); relative reduction=1-mean(recal_Delta)/mean(raw_Delta) if the raw mean is positive. Distinguish this ratio of means from mean per-cell R.

Per-cell outputs also include normal/off/recal NLL, fitted b, training rates, raw signed mean Delta logit and mean absolute Delta logit. Centered test logit difference is off-normal minus its scored-bin mean. Compute population-standard-deviation (ddof0) and RMS on the full scored vector in float64; they coincide up to numerical roundoff. These are descriptive temporal/stimulus-dependent variations under the frozen stimulus and observed-history convention, not an independent stimulus-only identification, new RF analysis or verdict rule.

Population: 22 biological cells, equal cell weight; within-cell scoring weights all valid bins as production. Raw/recal mean and median, recal positive/negative/exact-zero counts, 95% mean/median percentile intervals. Paired bootstrap100000 draws using numpy.default_rng(2026090502), frozen cell order from the prior zero results; reuse one22-column index matrix across pathways. Percentiles2.5/97.5 with linear interpolation. This new seed is statistical only and explicitly requested; no p-values, group tests, new exclusions or extra diagnostics.

Normal sanity: per-cell fitted b, train and test NLL before/after. Summarize its equal-cell mean/median test change and signs descriptively, without an added significance test or new primary model. Disclose material calibration drift instead of assuming normal is already exactly calibrated.

## Preregistered interpretation

Apply the user's four labels independently, without borrowing strength from another pathway. NO POSITIVE RESIDUAL SUPPORT when recalibrated mean Delta<=0 (report any improvement and heterogeneity). For the already-large direct-BC/AC raw penalties, OPERATING-POINT DOMINATED when more than half the population raw penalty is removed and the recalibrated mean interval includes zero. H1's small raw penalty is not relabelled as a large raw penalty; unresolved positive H1 residual is MIXED with explicit downgrade of quantitative predictive contribution.

For a positive residual with a mean interval entirely above zero and broad positive cell support, STIMULUS-DEPENDENT PREDICTIVE CONTRIBUTION SUPPORTED applies when scalar calibration removes less than half of the raw mean penalty; MIXED applies when calibration removes at least half while positive residual evidence remains, or cell support is heterogeneous. “Half” merely operationalizes majority/minority removal, not a significance threshold. Broad support is fixed before results as at least17 of22 cells having strictly positive recalibrated Delta (at least75 percent, rounded up); this operational sign-count rule is not a significance test. Other unresolved positive outcomes are MIXED. Centered-logit measures cannot change these verdicts. No post-result adjustment of these rules.

A remaining penalty only supports that the frozen intervention consequence is not explained solely by one global logit/rate shift. Removed penalty reflects response calibration under this fixed fit. Neither component establishes biological necessity, unique pathway causality, retrained structural ablation or a stimulus-only causal effect. The scalar is estimated entirely from training, while evaluation reuses the already consumed set. Preserve both facts and the prior evidence level.

## Outputs and stopping

Produce only requested per-cell fits, normal diagnostic, pathway per-cell/population and centered-logit tables, exact source identities/hashes, six-question REPORT plus one paper-level concluding sentence, and evidence_manifest.json. Preserve all previous artifacts. No automatic further experiments, robustness, group-meeting deck or other material creation is triggered by a positive result; the report/evidence are the requested handoff for subsequent paper/group-meeting organization. Stop on completion.
