# Independent final verification

Overall review: **PASS**. The five reviews were independent of the implementation and did not run new model forwards, training or experiments. Verification recomputed prescribed quantities from saved predictions and checked original source/checkpoint identities.

| Perspective | Reviewer | Verdict | Evidence |
|---|---|---|---|
| Goal and constraints | heldout_protocol_review | PASS | Seven requested answers, immutable protocol, clean-data gate before replay, 88 exact preflight passes, first-NLL consumption order, declared inference only, bounded interpretation. |
| Saved-output numerical QA | heldout_numeric_qa | PASS | 18,194 checks, zero failures. All 154 NLLs, 22 analytical Constants, 66 paired model deltas, 66 pathway deltas and 264 logit summaries reproduce exactly. |
| Implementation correctness | heldout_code_review | PASS | Production loader reset/mask/order, strictly-past binary history, grouped clamp semantics, batch8 original runtimes, equal-cell bootstrap and sole lexical-tie-broken deletion conform to protocol. |
| File and execution safety | heldout_safety_review | PASS | Independent SHA256 of all 2038 frozen inputs unchanged; local write boundaries; weights_only loads; fixed argv; no optimizer/backward/network/credentials/destructive actions; no output reparse points. |
| Prior context and lineage | heldout_context_review | PASS | All 88 final checkpoint hashes/refit metadata match; aligned22 additionally match prior analysis-manifest. Four development quantities match exact frozen source. Original CNN GPU-forward/CPU-NLL convention retained. |

Numerical QA checked all 22 saved cell files, 37 recordings, 137 trials, 5480 sequences and exact segments20..59 for each trial. The 822000 target bins divide into 657600 scored and 164400 masked. All 5480 source-range rows and 66 target/mask/order hash checks agree with saved tensors.

All six paired-bootstrap comparisons reproduced mean/median intervals and counts exactly using the frozen 100000 index draws, SHA256 `e7c2af0a9e6c1c54358f3f64513ad92790ea4bcc7647e92f2c9ab3a9da901528`. Independent centered-dot and explicit tie-rank correlation formulas differed from saved results by at most 2.220446049250313e-16. The prescribed float32 NLL calculation reproduced all losses exactly; an independent float64 formula differed by at most 7.644463728695428e-08, without altering any stored metric or pass threshold.

One report provenance link was corrected during review: RMS/full-vector norm columns are in pathway_per_cell.csv, while pathway_logit_vs_nll.csv contains the correlation input subset. The context reviewer confirmed the corrected link; no numerical values or protocol changed. Final output hashes were refreshed afterward.

The startup version-label and postprocessing compiled-dependency corrections are documented in startup_check_record.md. They did not trigger new held-out inference or modify the frozen scientific protocol. No blocking findings remain. This verification establishes the recorded execution and calculation, not biological pathway necessity.
