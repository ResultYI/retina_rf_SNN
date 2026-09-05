# Canonical V1 correctness patch — fixed acceptance scope

User request: repair only the four confirmed implementation-contract failures. Preserve H1 -> shared BC -> direct BC / broad BC -> AC; no training, physiology, loss, radius, temporal-bound, illusion, baseline, or revision-name changes.

Production edits are held until the lineage reference worker has captured all 22 raw-config, strict-loaded final checkpoints on the existing saved temporal and illusion inputs. Root source snapshot: source_before/ and source_before_manifest.json. The shared dirty worktree is preserved.

Independent responsibilities: lineage reference/candidate comparison; target support plus self-only mixing; custom geometry plus Canonical config/checkpoint validation. Root owns model.py and pathway_rf.py integration.

Confirmed hypotheses to reproduce before fixing:
1. Masking source kernels before cross-cell mixing leaves target kernels with a union of source supports. Distinguish with d(BC_direct/broad_i)/d(X_H1[c]) outside target disk; require exact zero outside and nonzero inside after patch.
2. Custom geometry shape/nesting checks do not enforce the full distance-defined disks. Distinguish with the original interior-hole mask and valid full-disk control.
3. Self-only rows normalize one positive trainable scalar to one for all parameter values. Distinguish optimizer membership and parameter perturbation on all-self-only and mixed-degree graphs.
4. The formal V1 validator and builder admit explicit legacy config; marker checks are registered only for mechanism mode. Distinguish explicit legacy configs/checkpoints under both strict settings and current checkpoint controls.

Fix constraints: target-mask the mixed kernels before cone contraction; keep N=1's original feature contraction then mixer operation order. Keep self-only rows as fixed identity; train only edges in rows with multiple neighbors. Validate custom geometry against the existing support constructor; keep default geometry unchanged. Require explicit Canonical mechanism/causal/spatial/schema identity at its entry points.

Mandatory acceptance: original counterexamples become green; existing relevant causality, exact-zero clamp, shared-parameter, no-bypass and RF/autograd/finite-difference checks remain green. No optimizer steps are permitted; constructing an optimizer only to inspect membership is allowed by the requested parameter check.

Lineage acceptance: all 22 strict-load with identical state keys, tensor values, shapes, dtypes and parameter/buffer roles, without checkpoint conversion. Compare all exposed normal output tensors and captured AC input on the same saved inputs. Any non-bitwise output difference blocks acceptance; do not choose a numerical tolerance after seeing it. Record the differing tensor/error and stop if it is substantive, as requested.

Artifacts are retained because the user requested evidence paths. No debug instrumentation is inserted into production. Only command-local no-bytecode/thread settings are used. This protocol replaces unrelated skill scaffolding/cleanup and approval gates; the user's explicit repair authorization and narrow scope control the task.
