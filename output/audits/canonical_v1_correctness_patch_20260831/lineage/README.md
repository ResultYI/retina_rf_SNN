# Current 22-cell lineage preservation evidence

Final result: **PASS**. All 22 checkpoints strict-load unchanged after the correctness patch. All 1,188 output arrays (300,920,400 scalar elements) are bitwise identical; maximum absolute error is exactly 0. Every state-dict tensor, parameter/buffer name, parameter trainability flag, and checkpoint/input file hash is unchanged. `SUMMARY.json` contains the aggregate result and `candidate/*.json` contains each tensor's comparison. Candidate per-cell time sums to 40.06728 seconds.

The pre-patch reference contains 22 strict-loaded final checkpoints, each evaluated once in normal mode on the saved temporal input bank selected by its recorded retinal class/polarity and once on the saved 72-stimulus illusion bank. Temporal history is all zero, matching the frozen temporal production protocol; illusion history is loaded unchanged from `illusion/inputs.pt`.

Each of the 44 reference NPZ files stores all 26 `MechanisticRetinaOutput` tensor fields and the actual AC input captured by a forward pre-hook: 1,188 tensor arrays in total. References occupy 205,480,151 bytes and the per-cell capture time sums to 47.5484332 seconds. All 22 models have 33 trainable scalar parameters and 52 state-dict tensors before the patch.

`reference_manifest.json` identifies the production source files, all 22 checkpoint files, and both saved input files. Per-cell JSON records retain strict-load status, raw checkpoint configuration, checkpoint hash, state tensor shape/dtype/raw-byte hash/parameter-or-buffer role, parameter and buffer names, and output hashes. Per-range integrity files verify no production or frozen-file changes occurred during reference capture.

`reference_trainability.json` supplements the original capture with each parameter's `requires_grad` flag, read from the preserved pre-patch source snapshot without any forward call. All imported mechanistic modules resolve inside that snapshot, and their 22 source files match the original reference source hashes. Each model has 129 named-parameter scalar elements: 33 trainable and 96 frozen operator elements. Candidate acceptance requires identical flags for every parameter, not merely an unchanged aggregate count. Summary also checks completed reference integrity coverage and exact preservation of all buffer names.

Run with the existing frozen runtime from the repository root:

```text
PYTHONDONTWRITEBYTECODE=1 D:/anaconda/python.exe -B output/audits/canonical_v1_correctness_patch_20260831/lineage/compare_lineage.py candidate 0 22
PYTHONDONTWRITEBYTECODE=1 D:/anaconda/python.exe -B output/audits/canonical_v1_correctness_patch_20260831/lineage/summarize_lineage.py
```

The comparator uses exact tensor bytes, including signed-zero representation. It rejects any changed state-dict identity or parameter count and stops on the first non-bitwise-identical output, writing `candidate/FAIL_STOP.json` with the tensor name and maximum absolute error. No tolerance permits a difference. Existing reference archives cannot be overwritten by the capture command. Candidate results are accepted only when every checkpoint completes and source/input/checkpoint integrity remains consistent across candidate ranges and at summary time.

No optimizer, training, checkpoint conversion, or checkpoint write is used. Normal-only output preservation is the scope of this artifact; focused production regression tests cover clamps separately.

An independent read-only harness review identified two evidence gaps before candidate execution: aggregate trainable counts did not prove individual trainability flags, and summary initially did not require completed reference-range integrity. The preserved-source supplement and complete reference-integrity checks address both. The reviewer rechecked the amended scripts and reported no remaining blocking finding. The original 44 reference archives were not regenerated or modified.
