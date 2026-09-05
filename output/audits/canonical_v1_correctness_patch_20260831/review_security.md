# Bounded trust-boundary review

Verdict: **PASS**. No blocking security or incompatible-load regression found within the four confirmed correctness failures and the introduced code.

Scope: read-only comparison of `canonical_contract.py`, `spatial_contract.py`, `pathway_spatial_geometry.py`, `shared_subunits.py`, `model.py`, and `karamanlis_v1_rf_validation.py` against `source_before/` (the canonical validator module is new). No code/test edits, tests, experiments, checkpoint conversion, or external data access were performed by this reviewer. This review does not assert comprehensive hostile-checkpoint validation.

Evidence:

- `models/mechanistic_retina/canonical_contract.py:12-20` and `model.py:70-76` reject explicit legacy architecture and incompatible causal/spatial identities before component construction. Both public construction routes reach this check.
- `evaluation/mechanistic_retina/karamanlis_v1_rf_validation.py:60-94` requires the existing schema/stage, current revision, complete config field set, Canonical config, and both uint8 state markers. Its only added catch translates `TypeError`/`ValueError` into a raised validation error with the original exception chained; failures are not suppressed.
- `pathway_spatial_geometry.py:33-74` checks shape, finite/nonnegative basis values, and exact equality of supplied BC/AC supports to the existing radius-defined disks. `bipolar_subunits.py:71-87` supplies those independently constructed supports.
- `spatial_contract.py:41-92` checks incoming identity, complete/equal support masks, basis shape/finiteness/nonnegativity, support pattern, and paired pathway equality. These checks raise from the root model pre-hook, before its state or child parameters are copied. The hook ignores PyTorch key strictness, so `strict=False` does not bypass the targeted checks. The existing causal pre-hook similarly rejects missing/legacy markers.
- `shared_subunits.py` adds fixed self-only mixing and removes its ineffective trainable coordinates; it adds no deserializer, file/network access, exception catch, or checkpoint conversion. The existing explicit-layout validation remains in place.
- The RF entry still uses `torch.load(..., map_location="cpu", weights_only=True)` at lines 102-104 and `load_state_dict(..., strict=True)` at line 127. The delta adds no unsafe loading fallback, network call, secret access, or broad exception suppression.

Recorded verification, inspected but not rerun: `regression_final.xml` has 101 tests, zero failures/errors/skips, including legacy and malformed support/path-basis rejection before mutation with both strict settings. `lineage/SUMMARY.json` reports 22/22 strict loads, unchanged state and trainability, no conversion, and 1,188 bitwise-identical tensor comparisons over the saved normal-mode inputs. These records establish the requested bounded acceptance evidence, not general checkpoint authenticity or biological validity.

Nonblocking scope limit: the new finite-value checks inspect the serialized tensor dtype. They do not add protection against arbitrary extreme finite values overflowing during an ordinary cross-dtype copy into a narrower destination. This generic pre-existing conversion behavior is outside the four confirmed failures and is not introduced by this patch; no additional guard, bound change, or experiment is requested by this review. Arbitrary trusted Python mutation of module attributes is likewise outside scope.

Reviewed source SHA256:

| File | SHA256 |
| --- | --- |
| canonical_contract.py | 344E87EC2544AC49C12EF704A20FC428C7A5F308F1F9BAACFCD425B62AEE1EBB |
| spatial_contract.py | 718923E8112175A2469460FB41D23CE974437A88670DB00EF75263DAF41A1A11 |
| pathway_spatial_geometry.py | DBD9C432668F6194CFC333778CFA178A2D2332230E9B4E11C571DC9E31ED9708 |
| shared_subunits.py | 78A9D7AF1C572851953BD8917F94E3149FA8823B33363FA04B903A575B25004D |
| model.py | 040B631F93AFF307D06F5D354D971EA9E9D00BF1EB5595A0ECBA1C2BE76828EF |
| karamanlis_v1_rf_validation.py | B43DCD7F8F180EEE0A59A294FF80A137E569043C1A5B1A18F390BB874B5DA6A1 |
