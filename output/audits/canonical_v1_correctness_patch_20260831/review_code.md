# Canonical V1 correctness patch: code review

Verdict: **PASS**. No blocking correctness issue found within the four requested fixes.

Review date: 2026-08-31. Production and test files were read only. The reviewer compared the nine assigned current files against `source_before/`, inspected the focused tests and completed lineage evidence, and used a second read-only reviewer for loader ordering and rejection semantics. No training, optimizer step, checkpoint conversion, production edit, or test edit was performed.

## Findings

No blocking findings.

## Verified implementation boundaries

- `models/mechanistic_retina/bipolar_subunits.py:163`: for N > 1, the source basis kernels are mixed first, then multiplied by the target BC/AC disk, then contracted with lagged cone input. The mask remains in the differentiable path. N = 1 retains contraction followed by the original feature mixer call order.
- `models/mechanistic_retina/shared_subunits.py:90`: raw trainable coordinates include only edges whose target row has multiple neighbors. All-self-only graphs have no mixer parameters; N = 1 keeps the existing one-element compatibility buffer but produces literal identity independently of its value. Mixed graphs preserve genuine mixing coefficients. At `:107`, row degrees and filtered mixing edges are derived from the loaded `edge_index`, so checkpoint edge order does not leave stale construction-time indexing.
- `models/mechanistic_retina/model.py:145` and `models/mechanistic_retina/pathway_rf.py:22`: production forward and the RF basis helper use the same corrected feature-bank route. Direct and broad BC still call the same `BipolarSubunits` instance; AC receives broad BC states and no independent stimulus input or encoder weights were introduced.
- `models/mechanistic_retina/pathway_spatial_geometry.py:69`: custom support values must equal the complete radius-defined disks from the existing constructor. `support_partition.py:39` rejects nonfinite coordinates without changing radii. Custom spatial basis values are retained.
- `models/mechanistic_retina/spatial_contract.py:28`: the root pre-hook checks markers and geometry before any retina state tensor is copied, including `strict=False`. Incoming BC/AC supports must match the full disks; basis shape, finiteness, nonnegativity, support pattern, and paired views are checked. Exact coefficient re-normalization is intentionally outside this repair: valid float32-computed buffers promoted to float64 must round-trip unchanged, and the requested defect is disk/support correctness.
- `models/mechanistic_retina/canonical_contract.py:13` and `model.py:70`: both public model construction routes reject explicit legacy mode. `evaluation/mechanistic_retina/karamanlis_v1_rf_validation.py:60` requires the current formal V1 schema, stage, revision, complete configuration, architecture, causal/spatial identities, and serialized marker tensors before evaluation loading.

## Runtime evidence inspected

The following completed artifacts were inspected; their runs were not repeated by this reviewer.

- `regression_final.xml`: 101 tests, zero failures/errors/skips. The retained cases cover target-support derivatives, true mixing gradients, optimizer membership, loaded edge order, invalid custom disks, strict and non-strict rejection before mutation, dtype-converted round-trip, shared BC/AC dependency, no AC bypass, structural clamps, and RF derivative agreement.
- `lineage/SUMMARY.json` and `lineage/README.md`: all 22 existing single-cell checkpoints strict-load; state identity, parameter trainability, and buffer names remain unchanged. Across the saved temporal and illusion inputs, all 1,188 output arrays (300,920,400 scalar elements, including captured AC input) are bitwise identical with maximum absolute error 0. No conversion or training was used.

This is a code-correctness review for the requested patch, not a biological-validity verdict or a general audit of arbitrary checkpoint corruption.

## Reviewed source hashes (SHA256)

```text
C14CFF98435E3D5401AC5307FE4AF080D04A7321F572ADEFCFDD1B04305751E5 models/mechanistic_retina/bipolar_subunits.py
78A9D7AF1C572851953BD8917F94E3149FA8823B33363FA04B903A575B25004D models/mechanistic_retina/shared_subunits.py
040B631F93AFF307D06F5D354D971EA9E9D00BF1EB5595A0ECBA1C2BE76828EF models/mechanistic_retina/model.py
211D3FDAAE9F5C0FAB17F2332098948BBD2A7C6A0D22CC46E1FAE17CA4DBA521 models/mechanistic_retina/pathway_rf.py
718923E8112175A2469460FB41D23CE974437A88670DB00EF75263DAF41A1A11 models/mechanistic_retina/spatial_contract.py
DBD9C432668F6194CFC333778CFA178A2D2332230E9B4E11C571DC9E31ED9708 models/mechanistic_retina/pathway_spatial_geometry.py
E049DE5E3AB44F5089936EAA94904EA2B27EF43D043169C4D6BADDAAFACC36B4 models/mechanistic_retina/support_partition.py
344E87EC2544AC49C12EF704A20FC428C7A5F308F1F9BAACFCD425B62AEE1EBB models/mechanistic_retina/canonical_contract.py
B43DCD7F8F180EEE0A59A294FF80A137E569043C1A5B1A18F390BB874B5DA6A1 evaluation/mechanistic_retina/karamanlis_v1_rf_validation.py
```
