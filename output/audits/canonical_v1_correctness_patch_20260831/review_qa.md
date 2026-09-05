# Independent no-training QA

Status: **PASS** for the requested regression scope.

The independent QA run executed only the three test files named in `PATCH_PROTOCOL.md` and the QA assignment. Pytest reported **60 passed, 0 failed, 0 errors, 0 skipped**, exit code **0**, in **4.54 s**. Run timestamp recorded by JUnit: `2026-08-31T12:08:23.434363` (local machine time).

## Executed command and evidence

Working directory: `D:/PythonProject/retina_rf_SNN`.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& 'D:/anaconda/python.exe' -B -m pytest -p no:cacheprovider -vv `
  tests/test_canonical_mixing_correctness.py `
  tests/test_canonical_geometry_entry_correctness.py `
  tests/test_causal_pathway_contract.py `
  --junitxml=output/audits/canonical_v1_correctness_patch_20260831/review_qa_pytest.xml
```

The actual invocation restored the three environment variables in `finally`; the settings were local to the command process. Runtime: Python 3.12.7, pytest 7.4.4, win32.

- Complete executed test names and terminal output: `review_qa_pytest.log`.
- Machine-readable result and individual timings: `review_qa_pytest.xml`.
- Tested file SHA256 values: `review_qa_test_hashes.json`. All three hashes match the values captured before execution.

## Acceptance mapping

| Required contract | Executed evidence | Observed result |
|---|---|---|
| Mixed direct/broad BC kernels obey each target disk | `test_target_support_derivative_is_zero_when_audited_six_cells_mix[direct/broad]`, mixing test lines 33-64 | Actual `forward_sequence` output is differentiated against the H1 forward-hook tensor. Each of 6 cells x 2 components has exactly zero derivative outside its corresponding BC/AC target support and nonzero derivative inside. All 24 assertions pairs pass. |
| Original disjoint-disk counterexample is repaired without removing its shared edge | `test_direct_support_is_local_when_two_nonoverlapping_bc_disks_mix`, lines 67-97 | Cross-cell connection remains positive; target direct BC derivative at the other cone is exactly zero and at its own cone is nonzero. |
| Custom geometry rejects holes/annulus/non-disk masks, preserves valid disks | Geometry test lines 42-80 | All four invalid mask fixtures are rejected; valid full-disk custom geometry retains the supplied spatial basis and supports exactly. |
| Checkpoint loading cannot introduce support/basis holes | `test_checkpoint_geometry_rejected_before_state_mutation`, lines 179-205 | Hole, basis-hole, basis-outside and missing-basis states are rejected under both `strict=False` and `strict=True`; every original state tensor remains unchanged. |
| Self-only rows have no ineffective trainable parameter | Mixing test lines 100-126 and 153-169 | All-self-only mixer has no parameters, its compatibility buffer is absent from optimizer groups, perturbing that buffer leaves literal identity and forward logits unchanged. The four-group model has 76 trainable scalars. N=1 retains the existing three-key state format and fixed identity. |
| Genuine mixed rows remain learnable | Mixing test lines 129-150, 172-189 and 191-201 | Mixed-degree fixture has four raw trainable coordinates rather than five; differentiated coefficient has two nonzero raw gradients; manual perturbation changes the mixing row while fixed identity row stays unchanged. Loaded edge reordering preserves behavior. Original six-cell audit fixture has 8 shared-mixer raw scalars and 86 trainable scalars total. |
| Canonical entry rejects explicit legacy mode | Geometry test lines 95-103 | Builder and direct model constructor both reject explicit legacy configuration. |
| Explicit V1 identity is required | Geometry test lines 106-149 and 165-176 | Nine incompatible serialized identity cases are rejected; complete current identity is accepted. |
| Legacy state is rejected before mutation, including `strict=False` | Geometry test lines 151-162; causal tests lines 177-219 | Both strict settings reject missing/wrong causal markers. Full tensor state remains unchanged, including nested-module load cases. |
| Existing causal graph, clamps and derivative contract remain intact | Entire `test_causal_pathway_contract.py` (22 cases) | Shared BC dependency, no independent AC stimulus encoder/bypass, shared encoder weights, exact-zero clamps, stimulus/history causality, global RF finite differences, pathway RF/autograd agreement, deterministic finite frozen forward and no carried state all pass. |

The production-support regression uses autograd through the real model forward; it does not infer support solely from stored masks. The no-training parameter checks distinguish optimizer membership, nonzero gradient and a manual perturbation. No claim of an optimizer update is made.

## Existing 22-checkpoint lineage evidence inspected

The separate lineage worker's completed `lineage/SUMMARY.json` and `lineage/README.md` were read without repeating any checkpoint forward. The summary reports PASS: 22/22 strict loads, unchanged state identity and per-parameter trainability, and 1,188 normal-output/AC-input array comparisons across the saved temporal and illusion inputs. All arrays are bitwise identical; maximum absolute error is 0.0. The reported preserved checkpoint/input hash count is 24. This paragraph attributes those results to that separate execution; the independent execution in this report is the 60-test run above.

## Scope and limits

No production or test file was modified by this reviewer. No optimizer step, training run, new training checkpoint, conversion, architecture change, extra experiment or additional test suite was performed. Only this report and its QA log/XML/hash evidence were written. These are implementation-contract results, not biological validation or scientific-performance claims.
