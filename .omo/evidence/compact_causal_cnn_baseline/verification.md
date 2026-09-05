# Focused compact causal CNN verification

## Contract tests

- Failing-first architecture existence check: 1 failed before implementation (module absent).
- Initial CNN suite: 7 passed; CNN plus inherited LN stopping/training suite: 14 passed.
- GPU future-input causality: PASS, PyTorch 2.10.0+cu126 / RTX 4070 Laptop GPU. Future bins 61..89 changed; logits 0..60 bitwise identical, later logits changed. TF32 off; deterministic CUDA enabled.
- Tested 60-bin stimulus gradient support, 2990 trainable parameters, identical LN history/bias head and strictly-past history shift, finite tiny-unit-fit gradients/updates, fresh initialization and validation-free selection signature.
- Six changed source/test files passed the scoped programming checker.

## Native data boundary

22 cells / 37 recordings; all 132 train/validation input, target and loss-mask digests match the corrected frozen SC preflight. Native loader uses PyTorch 2.6.0+cpu / NumPy 2.2.6. Input arrays are serialized without numerical conversion and consumed as identical float32 tensors in the GPU environment. Original LN history metadata and inner split are reused.

## Pre-training provenance completion

Read-only review found three directly used evaluation modules missing from the source list. Added their hashes and explicit preflight-identity guards before any CNN fitting. The initial preflight is retained in `preflight_before_provenance_completion.json`. Contract and all cell/data entries are unchanged; only source-guard metadata was completed.

## GPU entry-point dependency correction

The first launch failed during import, before `main`, optimizer creation, or any cell fit:
`ModuleNotFoundError: No module named 'cv2'`.

Hypotheses checked:

1. CNN/CUDA computation is broken: rejected; exact GPU causality and native-cell forward had already passed.
2. Wrong interpreter or corrupt input: rejected; observed interpreter is `D:\anaconda\envs\snn_env\python.exe`; `cnn_training_import PASS`; input bundles had already matched native hashes. No input was read by the failed entry point.
3. Hash-helper import unnecessarily pulls in the movie decoder: confirmed by the traceback `schottdorf_multirecording_reporting -> schottdorf_multirecording_types -> schottdorf_lee_2021 -> cv2`; `find_spec('cv2') = None`.

The new runner's hash helper is replaced by standard-library SHA256, without importing data decoders or installing packages. The failing-first GPU runner-import test reproduces the same error before this edit. No architecture, initialization, optimizer, split, data tensor, or numerical training function changed. No debugger/listener/instrumentation or temporary debug files were created; startup failure output is retained as benchmark provenance.

After the import fix: `GPU runner import PASS`; focused CNN plus LN suite `15 passed in 2.79s`. Runner/test source hashes were refreshed before the first optimizer step. The original failing launch and failing test performed zero data fits.

## Scoped implementation review

| Lane | Final verdict | Scope |
|---|---|---|
| Goal/constraints | PASS | Fixed CNN/head, exact inner-dev/refit contract, original validation isolation, native data and comparison sources |
| QA | PASS | Focused tests and exact GPU future-input causality; final entry import regression also passed |
| Code correctness | PASS | CNN/trainer/report path; follow-up approved the standard-library SHA256 entry fix |
| Artifact safety | PASS after correction | Output confinement, weights-only loading, added dependency hashes and preflight guards |
| Source context | PASS after correction | 22 cells/37 recordings, native tensor identities, final corrected SC and saved shared-BC comparison values |

Reviewers did not train data fits or modify sources. Historical shared-BC predictions are used as the frozen Canonical reference; there is no claim that its historical model-source tree equals today's tree. The immutable-source check refers to this CNN run.

## Completed benchmark execution

- Actual run exited 0: 22 cells, 44 inner-development fits and 22 fresh full-train refits. No data fit was restarted.
- Learning-rate selection: 16 cells selected 0.001; 6 selected 0.0003. All other numerical training settings stayed fixed.
- Every refit reports total/requires-grad/optimizer-listed parameter count 2990 and finite gradients. All saved validation outputs are finite.
- Report regenerated NLL using the native CPU evaluation runtime. Maximum absolute GPU/native-CPU NLL difference: 5.960464477539063e-08. Reported comparison uses the CPU values consistently.
- All 22 saved validation target/mask/order identities passed exact comparison. All frozen source and baseline-artifact hashes remained unchanged through fitting and reporting.
- Final CNN equal-cell mean validation NLL: 0.42259781062602997. Full per-cell and four-group comparisons are in `per_cell.csv`, `group_summary.csv`, `results.json`, and `summary.md`.
- One non-fatal PyTorch warning came from converting the already-computed training loss to a scalar for progress logging. No numerical or optimizer change was made in response.
- Final scoped source checker: `no violations in 6 file(s)`. Existing tracked worktree changes were preserved. No packages were installed, no Canonical/LN/SC source or artifact was changed, and no additional baseline was started.

## Final read-only artifact arithmetic review

- PASS: independently checked 22 cell records, 44 inner fits and 22 refits; no additional fits, model loading, or test-suite runs.
- All selected learning rates minimize inner-dev NLL; recorded best steps equal the first curve minima and full-train refit steps. Each of the 44 curves contains `stopping_step + 1` entries.
- All parameter-count records are 2990 total / requires-grad / optimizer-listed. LR selections are 16 at 0.001 and 6 at 0.0003.
- CSV-to-JSON maximum discrepancy is 0. Independent recomputation of five group rows by five NLL metrics has maximum discrepancy 0.
- Runtime/preflight hashes match. Original validation is identity-checked before fitting, evaluated after refit, and excluded from selection.
