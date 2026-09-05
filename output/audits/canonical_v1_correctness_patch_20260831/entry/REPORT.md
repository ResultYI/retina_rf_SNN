# Custom geometry and Canonical entry correctness patch

Scope: architecture-audit findings 2 and 4 only. No training, checkpoint conversion, parameter changes, radius changes, or physiology changes were performed by this work unit.

## Implementation

- `models/mechanistic_retina/pathway_spatial_geometry.py`: custom BC and AC masks must numerically equal the complete radius-defined disks produced from the unchanged cell-type radii. The existing finite/nonnegative basis and nonempty strict-subset checks remain.
- `models/mechanistic_retina/support_partition.py`: rejects nonfinite positions at the support-construction boundary.
- `models/mechanistic_retina/spatial_contract.py`: the root load pre-hook also validates incoming geometry before PyTorch copies any state. Incoming supports must match constructor disks; spatial buffers must have correct shapes, finite nonnegative values, matching allowed nonzero patterns, and identical repeated direct/broad spatial views. Missing spatial buffers are rejected even with `strict=False`.
- `models/mechanistic_retina/canonical_contract.py`: requires `mechanism_identifiable`, the current shared-BC causal contract, and the current overlapping full-disk contract. Root-owned `model.py` calls this before constructing model components. The generic config dataclass retains its legacy representation; no silent conversion occurs.
- `evaluation/mechanistic_retina/karamanlis_v1_rf_validation.py`: retains its Karamanlis schema/stage/gain checks and additionally requires revision 4, the exact current config field set, valid current canonical config, and both existing serialized contract markers.
- `tests/test_canonical_geometry_entry_correctness.py`: regression coverage for construction, checkpoint loading, serialized identity, and valid dtype-converted roundtrip.

The `bipolar_subunits.py` call-site change is owned by the mixing work unit: it builds the unchanged disk partition once and supplies it to the custom-geometry validator.

Load validation checks geometry semantics without reconstructing normalized floating-point weights. This preserves ordinary float32-to-float64 module conversion, where re-normalizing the already converted basis would introduce a different rounding order. It neither accepts a tolerance nor mutates incoming values.

## Regression evidence

Initial pre-production-patch run of the new entry tests: **13 failed, 8 passed** in 1.55 s. All 13 failures were missing expected rejection; no import/setup failures. `red_reference.json` records the cases.

A subsequent independent review identified that a current-marker checkpoint could overwrite constructor geometry. Before adding the load guard, the targeted run produced **6 failed, 1 passed** in 1.50 s: hole masks, outside-support path basis, and missing path basis under both strict settings. The strict missing-key case also showed state mutation before ordinary PyTorch rejection. Raw tool output is retained in `geometry_load_red_tool_result.json`.

Final command:

```text
PYTHONDONTWRITEBYTECODE=1 D:/anaconda/python.exe -B -m pytest tests/test_canonical_geometry_entry_correctness.py tests/test_spatial_checkpoint_contract.py tests/test_karamanlis_v1_rf_validation.py -q -p no:cacheprovider
```

Result: **42 passed** in 1.77 s; raw tool output is in `green_tool_result.json`. This includes 30 new cases and 12 existing cases, with no training.

The final cases include interior-hole/contour/annular/nonbinary custom masks; public builder and direct constructor rejection of legacy; wrong/missing config identities and revision; missing causal/spatial markers; current-marker hole masks, path-only holes, outside path support, and missing path basis under `strict=True/False`; rejection before state mutation; and bitwise state roundtrip after `.double()`.

## Frozen 22-checkpoint config compatibility

All 22 saved final checkpoints have the exact current config field set, revision 4, current mode/contracts, and valid existing state contract markers. Their actual schema remains `schottdorf_canonical_v1_shared_bc_development`; it was not relabeled as the Karamanlis schema. Evidence: `checkpoint_config_identity.json`.

This work-unit check performs no inference and does not replace root-owned before/after tensor comparison or strict-load evidence. It creates no persistent model-state keys and changes no trainable parameter counts.

All six owned source/test files passed Python syntax parsing; hashes and line counts are in `source_checks.json`. The existing RF-validation module is 235 nonblank/noncomment lines; no unrelated refactor was performed.
