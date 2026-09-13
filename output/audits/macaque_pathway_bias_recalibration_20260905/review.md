# Independent verification

Overall **PASS**. All five required perspectives were independently reviewed without new forwards, fits, training or writes by reviewers.

| Perspective | Reviewer | Verdict | Evidence |
|---|---|---|---|
| Protocol / goal / constraints | heldout_protocol_review | PASS | Broad support was made deterministic before freezing; six-question report, train-only scalar estimation, consumed-set status and prescribed verdicts comply. |
| Numerical QA | heldout_numeric_qa | PASS | 7922 checks, zero failures; direct derivative/convex objective, raw/recal NLL, ratios, centered components and bootstrap recomputed from saved tensors. |
| Code correctness | heldout_code_review | PASS | Exact replay before fitting, training-only file separation, valid bracket/residual solver, fit hash lock, original-normal reference and fixed statistics. |
| Safety / write boundaries | heldout_safety_review | PASS | All2120 original/prior-artifact hashes unchanged; writes stay in the new directory, no model updates/optimizer/network. |
| Prior context / lineage / claims | heldout_context_review | PASS | Same22 aligned checkpoints, original contracts and consumed [20,60) evaluation; no raised evidence level or biological-necessity claim. |

Numerical verification: all88 saved biases satisfy the independently evaluated direct gradient mean(sigmoid(z+b)-y), maximum absolute residual9.970582560639298e-13 (NumPy9.97056405692222e-13). This is below1e-12; tiny differences from the recorded mean(sigmoid)-mean(y) residual arise from floating-point reduction order. All88 brackets contain b and bracket the derivative; all88 training objectives decrease. The792 saved fit-field comparisons are exact; independent NumPy likelihood differs by at most3.3306690738754696e-16.

All22 normal sanity rows and66 pathway rows reproduce exactly, including raw/recal NLL and deltas. The63 defined per-cell ratios are un-clipped: H1 has10 below0 and3 above1; direct-BC has11 below0; AC has6 below0. Three H1 ratios are undefined because raw penalty<=0. All264 centered-component fields reproduce exactly; independent ddof0 std/RMS differences are at most3.108624468950438e-14.

All three population summaries and24 mean/median CI comparisons across CSV/JSON reproduce exactly. Bootstrap seed2026090502, draws100000, index SHA2561202c559aa73a42663a9a35ccce678d7ee23f676c05904cd9d283ab1ed024232. Training covers137 trials,2192 sequences,263040 scored bins with segments0..15; test covers5480 sequences,657600 scored bins with segments20..59. All88 original float32 NLL and logit-hash checks are exact.

Safety independently rehashed2038 original inputs plus82 preceding held-out artifacts, all unchanged. The18 output hashes present at that review matched; final manifest was refreshed after adding this review record. No material findings remain. No further experiments were run after completing the specified control.
