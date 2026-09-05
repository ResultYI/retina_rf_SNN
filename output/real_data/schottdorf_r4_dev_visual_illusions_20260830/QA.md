# Frozen-evaluation verification

No training, optimizer, backward pass, production-model edit, or checkpoint write was performed.

| Review | Verdict |
|---|---|
| Requested probes, outputs and constraints | PASS |
| Independent saved-tensor/CSV QA | PASS |
| Indexing, metrics and aggregation | PASS |
| Local file security and provenance | PASS |
| L+M, geometry, time and history contract | PASS |

- Stimulus contract tests: 3 passed. Five artifact Python files parse; four evaluation modules import.
- 22 cells, 3 modes, 63 stimulus sequences (including controls and blank), 150 bins.
- All saved response tensors finite; sigmoid(logit) equals saved probability exactly.
- Target luminance matching passes for all 31 pair definitions. Every contextual control A-B response is bitwise zero.
- Saved AC-off local/transient currents have zero nonzero elements. Saved observed-spike history is zero.
- Independently recalculated 24,816 per-cell mean-on values, 5,640 group mean-on values and 528 Mach-boundary rows: maximum error zero.
- All seven response metrics for the three contextual pairs in cell 67#4, in all modes, independently agree.
- All 50 recorded source/checkpoint SHA256 hashes agree with current files.
- H1 exact-zero, in-memory state_dict equality, absent gradients and normal re-entry equality were asserted during each of the 22 frozen inference runs. H1 currents were not retained in the output tensor file.
- The 15 active bins have envelope 0.9999985098838806, exactly from the existing float32 bin-overlap function; saved cone drive exactly equals patches times this envelope.
- Rendered stimulus/control, population response and grouped Mach temporal figures were visually checked. There are 29 PNG figures including 22 per-cell figures.

Input is an ideal relative L+M luminance probe in the dataset adapter's Weber representation, not an absolute-display photometric calibration. All predictions condition on zero observed spike history. No perceptual decoder or biological inference was added.
