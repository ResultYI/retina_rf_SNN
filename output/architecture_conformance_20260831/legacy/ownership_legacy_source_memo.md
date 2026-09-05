# Ownership / legacy source memo

Audit date: 2026-08-31 (Asia/Tokyo). Repository: D:/PythonProject/retina_rf_SNN.

## Scope and independence

This memo records the ownership, intervention, and legacy-reachability sub-audit for the user's Architecture Conformance Audit. The pasted mathematical contract is the only normative specification. Evidence came from production source and the in-memory probe saved alongside this memo. No RESULTS.md, verification.json, CURRENT_STATE.md, historical audit conclusion, or existing checkpoint was read. No production source or checkpoint was modified and no training was run.

The probe was executed once before these artifacts were saved. Saving did not rerun the probe. Its original exit code was 0; timedOut was false. The shell invocation is preserved in legacy_in_memory_probe.sh. legacy_in_memory_stdout.json contains its original stdout JSON with line endings normalized. Neither a checkpoint file nor a state_dict file is included.

The root audit owns the main Canonical runtime identity, dependency, and spatial tests. This memo does not substitute static reuse for runtime object/storage/optimizer identity measurements.

## Parameter ownership (explicit mechanism_identifiable mode)

N = cell count; G = distinct (cell_type, polarity) groups; E = shared-subunit edges.

| Parameter / state_dict key | Shape | Owner | Source definition / use |
|---|---|---|---|
| h1.raw_tau | scalar | H1 | models/mechanistic_retina/h1_pathway.py:49-65 |
| h1.raw_delay | scalar | H1 | h1_pathway.py:53-69 |
| gates.raw_h1_amplitude | scalar | H1 | pathway_gates.py:60-76; h1_pathway.py:86 |
| feature_bank.raw_tau | [2,3] | shared BC encoder | bipolar_subunits.py:119,137-148,162-174 |
| feature_bank.raw_delay | [2] | shared BC encoder | bipolar_subunits.py:128,150-168 |
| shared_subunits.raw_connections | [E] | shared BC encoder | shared_subunits.py:84-107; N=1 is a buffer |
| bipolar.raw_weights | [G,2,2,3] | shared BC encoder | bipolar_subunits.py:186-207 |
| amacrine.raw_tau | [2] | AC downstream dynamics | amacrine_pathways.py:38-40,50-70 |
| amacrine.raw_delay | [2] | AC downstream dynamics | amacrine_pathways.py:43-46,58-70 |
| gates.ac_local | [G] | AC downstream pathway mixture | pathway_gates.py:65,87-95 |
| gates.ac_transient | [G] | AC downstream pathway mixture | pathway_gates.py:66,87-95 |
| gates.history | scalar | RGC / history | pathway_gates.py:67,96; rgc_state.py:66-75 |
| rgc.response_bias | [N] | RGC / readout | rgc_state.py:29,71-75 |
| optional cell_gains.log_bc | [N] | direct BC readout / gain | cell_specific_gains.py:22-55 |
| optional cell_gains.log_ac | [N] | AC downstream gain | cell_specific_gains.py:23-55 |
| alternative cell_gains.log_bc_sustained | [N] | direct BC readout / gain | cell_specific_gains.py:66-98 |
| alternative cell_gains.log_bc_transient | [N] | direct BC readout / gain | cell_specific_gains.py:67-98 |
| alternative cell_gains.log_ac_local | [N] | AC downstream gain | cell_specific_gains.py:68-98 |
| alternative cell_gains.log_ac_transient | [N] | AC downstream gain | cell_specific_gains.py:69-98 |

The two cell-gain modes are mutually exclusive (contracts.py:103-104). With N>1, the baseline mechanism-mode trainable element count is 16 + 14G + N + E; aggregate gains add 2N and pathway-mixture gains add 4N. With N=1, raw_connections is a non-trainable buffer and contributes no trainable element.

operator.depthwise.weight and operator.depthwise.bias remain nn.Parameters but requires_grad=False (neural_operators.py:11-21). They are absent from the production optimizer. Canonical mode rejects operators_enabled=True (model.py:139-140), and disabled forward returns ones (neural_operators.py:24-27).

### Source identity and optimizer construction

- model.py:145-150 calls the same self.bipolar object for both direct and broad views.
- bipolar_subunits.py:162-168 repeats a temporal basis produced from one raw_tau/raw_delay pair; it does not create a second encoder Parameter.
- bipolar_subunits.py:197-207 derives both calls' weights from self.raw_weights.
- amacrine_pathways.py:38-46 declares only downstream tau/delay; its stimulus-bearing input is bc_presynaptic (62-84).
- training/mechanistic_retina/optimizer.py:17-33 lists the original model Parameter objects and optional gain raw_parameters, filtering requires_grad. It does not clone or reconstruct them.
- Minimal optimizer construction is build_phase1_optimizer(model, learning_rate=1e-3); its implementation is Adam(phase1_parameters(model), lr=learning_rate) at optimizer.py:36-41.
- trainer.py:128-130 constructs that optimizer and trainer.py:141-155 consumes real forward logits. sampled.py, real_sampled.py, real_early_stopping.py, and r4_development.py also call this optimizer builder.

### Structural ineffective-coordinate boundary

shared_subunits.py:90-93 only special-cases N=1. For N>1, raw_connections is trainable in mechanism mode. connection_matrix (97-104) divides each row's positive weights by its sum. A row with only its self edge therefore has coefficient softplus(r)/softplus(r)=1 independent of r. If every row is self-only, the whole raw_connections tensor is structurally ineffective. If only some rows are self-only, their coordinates are ineffective even when the tensor has other nonzero gradients. This memo establishes the source algebra; it did not run an additional self-only probe.

An all-zero observed history makes gates.history have zero gradient for that input (rgc_state.py:66-75); that is an input-dependent zero, not proof of a structurally dead parameter.

## Intervention source trace

Static C verdict: PASS, to be combined with the root audit's measured runtime matrix.

| Contract | Source-based observation |
|---|---|
| H1-off amplitude only | pathway_gates.py:87-109 zeroes only the effective selected gate. h1_pathway.py:78-87 still computes graph, delay, low-pass; surround = amplitude * graph.transpose_apply(state), X_H1 = X - surround. |
| H1-off recomputes downstream | model.py:141-155 always computes feature bank, shared mixer, direct/broad BipolarSubunits calls, and AC. |
| AC-off contribution only | amacrine_pathways.py:79-84 computes states before multiplying by the negative local/transient gate. BC/H1 do not consume those AC gates. |
| No clamp renormalization | pathway_gates.py:88-96 computes softmax before individual zeroing; other pathway gate values are not renormalized. |
| direct-BC-off preserves broad and AC | model.py:148-155 computes broad/AC before the direct-only mask at 156-164. |
| Current clamp reaches true logits | model.py:160-174 creates total from clamped currents and calls RGC. rgc_state.py:62-76 produces logits from that total. |
| Gains do not mix pathways | cell_specific_gains.py:54-55 and 97-98 are componentwise multiplication. |
| Old ambiguous BC strings rejected | contracts.py:22-29 exposes only direct-BC clamp names; model.py:138 converts every supplied value to PathwayClamp. |

Formal counterfactual callers include evaluation/mechanistic_retina/schottdorf_fresh_evaluation.py:46-54,135-160 and clean_sampled_benchmark.py:234,260-272. rf_effective.py:16-29 differentiates real forward logits.

## Legacy reachability

### No independent AC encoder found in current model

The explicit mechanism_identifiable forward, and also the current legacy branch, use model.py:143-155: feature bank -> shared mixer -> the same BipolarSubunits twice -> bc_broad -> AmacrinePathways. The legacy finding below must not be described as an independent AC stimulus bypass.

The retained PathwayLocalOperator is frozen and disabled in Canonical mode. models/cells/amacrine.py belongs to the separate models/response_snn.py:73-85 call chain; the mechanistic Canonical model does not import that implementation.

The standard support builder uses overlapping full disks (support_partition.py:48-55,62-69), not an exclusive annulus. Annuli found in mechanism_teacher_support.py:97-120 and schottdorf_temporal_center_surround.py:92-116 define stimulus probes rather than encoder support.

### Confirmed V1 validator/build path accepts legacy

The formal validator evaluation/mechanistic_retina/karamanlis_v1_rf_validation.py:56-71 checks only schema, stage, Mapping type, aggregate gains enabled, and pathway-mixture gains disabled. It does not check architecture_mode.

The formal load sequence at 90-104 adopts checkpoint model_config.architecture_mode, constructs MechanisticRetinaConfig and build_mechanistic_retina, then strict-loads model state. karamanlis_v1_ac_perturbation.py:102-126 uses the same validation boundary and similarly adopts the mode. model.py:70-81 only registers causal/spatial load hooks for mechanism_identifiable mode.

Legacy PathFeatureBank uses spatial[:, None].expand(...).clone() at bipolar_subunits.py:113-114 without applying BC/AC support masks. It thus uses the same actual spatial basis for direct/broad while retaining distinct support buffers. It also uses per-cell BC grouping, radius-zero/frozen shared mixing, and frozen gates; it does not restore a separate AC encoder.

The probe independently observed:

| Observation | Value |
|---|---|
| Fixture | in-memory only, no checkpoint I/O |
| Seed / torch / dtype | 319 / 2.6.0+cpu / torch.float64 |
| Input | [1,32,49], 2 midget-ON cells |
| Minimal validator accepts legacy | true |
| Full in-memory model payload validator accepts legacy | true |
| Loaded mode | legacy |
| state_dict key count | 50 |
| Strict load missing / unexpected | [] / [] |
| Root load pre-hook count | 0 |
| Forward logits shape / finite | [1,32,2] / true |
| Reloaded vs source logits max absolute difference | 0.0 |
| Direct/broad actual path_spatial_basis equality | true |
| Direct/broad presynaptic output equality / maximum difference | true / 0.0 |
| BC/AC support buffer equality | false |

I verdict for this entry boundary: FAIL. A legacy model can satisfy the V1 validator and execute the loader's build/strict-state-load/production-forward segment with direct and broad views identical. The in-memory fixture did not execute the full filesystem/data-loading evaluation workflow. This does not establish that any existing real checkpoint currently uses legacy.

### Other legacy and geometry boundaries

- contracts.py:44 defaults architecture_mode to LEGACY. Current real-data construction sites explicitly choose mechanism_identifiable: schottdorf_multirecording_fit.py:49-62, schottdorf_real_run.py:52-63, karamanlis_rf_population_run.py:83-97, mechanism_runtime.py:85-99.
- mechanism_protocol.py:85-94 calls a legacy diagnosis model separately from Canonical student creation; this alone is not evidence of Canonical contamination.
- training/mechanistic_retina/stages.py:146-155 and direct_model_eval.py:63-75 retain legacy-default old APIs. No external production .py callers for run_seed_stage/run_mechanistic_sampled were found in the scoped scan; they are not claimed absolutely unreachable from external code.
- The shared mixer is after support application (model.py:143; shared_subunits.py:106-107), so non-self edges can import neighbors' cone supports. The root spatial audit owns the numerical test.
- data/karamanlis_rf_artifact.py:121-159 constructs AC support from the sampled RF contour and BC support from contour intersected with a radius. pathway_spatial_geometry.py:52-65 validates strict containment, not a circular disk. This is separate from the exclusive-annulus check and was passed to the root spatial audit.

## Artifact inventory

- legacy_in_memory_probe.sh: exact shell command used for the already-completed probe.
- legacy_in_memory_stdout.json: original numeric stdout, no checkpoint/state_dict payload.
- ownership_legacy_source_memo.md: this source memo.
