# Current research state — independent audit handoff, 2026-09-05

## Objective and current contract

The goal is real visual stimulus -> species/experiment-specific cone front-end -> physiologically constrained retinal dynamics -> RGC spikes, and to determine which RF, pathway contributions and temporal dynamics are identifiable from RGC responses. Prediction, model-internal intervention and biological identification are separate endpoints.

Public model: **Canonical V1**. Causal identity: `h1-shared-bc-direct-broad-ac`. Spatial identity: `bc-central-disk_ac-overlapping-full-disk`. One BC encoder supplies narrow direct and broad AC-presynaptic views. AC is downstream of BC. Historical `revision: 4` alone is insufficient to identify a compatible checkpoint.

Tau, explicit pathway delay (ms), RF lag window (bins) and strictly-past RGC history shift (one native bin) remain distinct. The usual 16-lag macaque Jacobian has a 100 ms lag-grid span; it is truncated, not automatically a complete dynamic or spike-triggered RF.

## Final macaque evidence

- 22 cells / 37 recordings: MC ON 5, MC OFF 4, PC ON 9, PC OFF 4.
- Final fits: `output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/`.
- Native 150 Hz; per trial first 16 s train, next 4 s validation; 150-bin sequences, 30-bin warmup.
- Inner selection: per-training-trial 80/20 with 60-bin guard. Adam LR=0.03, batch=4, max=1000, patience=200, min_delta=1e-7, no regularizer. Fresh full-train refit for selected steps.
- Existing 22-cell mean NLL: Constant 0.509817266, LN 0.425997944, CNN 0.422597811, Canonical 0.438956146, SC-adapted 0.458313940.
- Lower parameter count is neither matched-capacity evidence nor proof of physiological identification.

## Limits retained for independent review

1. Acquisition t=0 versus decoded frame 750/751 remains `UNRESOLVED_EXTERNAL_EVIDENCE_REQUIRED`. No timing change was made in cleanup.
2. Bernoulli conversion discards 15.901100% of counted spikes in combined scored bins; MC ON loses 30.129494%. Boolean target consistency passed on 328,800 bins. This is information loss, not a demonstrated target mismatch.
3. Reset/pre-roll sensitivity is small on the same 65,760 validation bins; that does not establish dynamic-state identification.
4. Independent-seed evidence covers four representative cells, two additional fits each. The 81 combinations hold 18 primary fits fixed; neither is a full 22-cell multi-seed experiment.
5. Parametric SBC/Mach uses shared zero history and paired logit over 300–400 ms. These are model outputs, not perceptual observations.
6. The 14/8 AC-effect sign split is exactly ON/OFF. Raw Mach reversal depends on cohort composition. SBC aggregate reversal survives balancing although four individual group means do not reverse.
7. Cell bootstrap CIs condition on frozen fits and are pointwise; they do not establish animal-independent replication, training-seed uncertainty or simultaneous surface confidence.
8. Current synthetic shared-BC noise-free recovery is a same-family controlled check, not biological hidden-state validation. Older independent-AC results do not substitute for it.
9. Marmoset RF geometry and 60-cell results are retained as a separate historical lineage, not current macaque validation.
10. Historical architecture/applicability audits predate a correctness patch. Read their source hashes and subsequent verification together; do not promote a historical PASS/FAIL to a universal current assertion.

Use [AUDIT_INDEX.md](AUDIT_INDEX.md) for evidence. Cleanup changes navigation, publication rules and superseded artifacts only. It does not choose an architecture or claim publication readiness; that judgment is requested in `audit/PRO_AUDIT_PROMPT_ZH.md`.
