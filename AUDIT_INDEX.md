# Independent audit entry point

This is a navigation map. Verify statements against source and numerical artifacts.
Old absolute Windows paths in immutable manifests map by repository-relative suffix.
Do not rewrite historical source hashes to match the current working tree.

## Evidence order

| Topic | Evidence |
|---|---|
| Final prediction, paired bootstrap, all per-cell NLLs | [.omo/evidence/final_prediction_results/README.md](.omo/evidence/final_prediction_results/README.md) |
| Current fits, configs and selection trajectories | [shared-BC results.json](output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/results.json); `cells/<cell>/model-trained.pt`, `inner-trajectory.csv`, `refit-trajectory.csv` |
| Learned parameters, RF and interventions | [shared-BC RESULTS.md](output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/RESULTS.md); `rf-tensors.pt`, `perturbation-tensors.pt`, `learned-pathway-quantities.csv` |
| Frozen applications and White/Hermann diagnostic/controls | [application RESULTS.md](output/real_data/schottdorf_canonical_v1_shared_bc_frozen_applications_20260830/RESULTS.md); `illusion/`, `temporal/`, `input-manifest.json`, `verification.json` |
| Parametric SBC/Mach, Canonical/LN/CNN | [parametric summary](.omo/evidence/parametric_illusion_benchmark/summary.md); `protocol.json`, `per_cell_curves.csv`, `responses.pt`, `stimuli.pt` |
| Required qualification of cohort reversal | [aggregation check](.omo/evidence/parametric_illusion_benchmark/aggregation_check/summary.md); four-group curves, labeled surfaces, control-subtracted interaction |
| LN checkpoint/filter provenance | [LN results](output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/results.json); `cells/<cell>/ln-trained.pt` |
| CNN fitting and predictions | [CNN summary](.omo/evidence/compact_causal_cnn_baseline/summary.md); `preflight.json`, `results.json`, `cells/` |
| SC-adapted, w=0 control, 69#6 correction | [SC summary](.omo/evidence/spatial_contrast_adapted/summary.md); `correction_69_6_a0_4/`; [official source manifest](.omo/evidence/spatial_contrast_baseline/source_manifest.json) |
| Fresh current-contract synthetic recovery | [synthetic results](output/synthetic_canonical_v1_shared_bc_noise_free_3seeds_20260830/results.json); teacher, student raw/trained checkpoints and RF/counterfactual tensors |
| Four-cell independent-seed evidence | [seed summary](.omo/evidence/real_data_independent_seed_sanity/summary.md); selection/seed records, `aggregate_summary.md`, `aggregate_81_combinations.csv` |
| Timing and remaining external blocker | [timing report](.omo/evidence/schottdorf_lee_timing_contract_final_check.md); [frame-zero unresolved report](.omo/evidence/schottdorf_lee_frame_zero_resolution.md) |
| Multi-spike binning and target consistency | [count sanity](.omo/evidence/schottdorf_lee_150hz_multispike_sanity/summary.md); `production_target_consistency.json` |
| Reset and causal pre-roll | [reset sensitivity](.omo/evidence/schottdorf_lee_reset_preroll_sensitivity/summary.md) |
| Correctness and history semantics | [correctness patch](output/audits/canonical_v1_correctness_patch_20260831/REPORT.md); [history audit](output/audits/illusion_history_semantics_20260831/REPORT.md); `output/architecture_conformance_20260831/`, `output/audits/current_checkpoint_applicability_20260831/` |

## Source entry points

| Contract | Files |
|---|---|
| Actual forward and defaults | `models/mechanistic_retina/model.py`, `contracts.py`, `canonical_contract.py` |
| Shared BC and downstream AC | `bipolar_subunits.py`, `pathway_temporal.py`, `amacrine_pathways.py` in that model directory |
| H1, mixtures, gains, support, tau/delay | `h1_pathway.py`, `pathway_gates.py`, `cell_specific_gains.py`, `support_partition.py`, `spatial_contract.py`, `temporal_parameters.py`, `delay_parameters.py` |
| RGC and observed history | `rgc_state.py`, `state.py` |
| Raw spikes, native time, metadata, split | `data/schottdorf_lee_catalog.py`, `schottdorf_lee_spikes.py`, `schottdorf_lee_multirecording.py` |
| Actual final trainer | `training/mechanistic_retina/r4_development.py`, `optimizer.py`, `losses.py`; final fit directory `run.py` |
| Shared inner split | `evaluation/mechanistic_retina/factorized_ln_split.py` |
| RF and structural intervention | `evaluation/mechanistic_retina/rf_effective.py`, `pathway_decomposition.py`, `structural_ablation.py`; artifact-specific producers |
| Baselines | `baselines/center_surround_ln.py`, `compact_causal_cnn.py`, `spatial_contrast_adapted.py`, `spatial_contrast_official.py`; corresponding training modules |

## Historical and local-only dependencies

Older R4/overlapping-support JSONs supply frozen cell order, split/config and
comparators to current producers. Their retention prevents broken provenance;
their checkpoints are not current Canonical models. Marmoset, earlier synthetic
and readout modules are historical dependencies, not newly endorsed baselines.

Check [DATA_AVAILABILITY.md](audit/DATA_AVAILABILITY.md),
`audit/cleanup_20260905/local_only_artifacts.csv` and
`audit/cleanup_20260905/publish_manifest.csv` before claiming an artifact was
available to the reviewer. Missing binaries can prevent independent recomputation
even when derived CSVs exist. The cleanup ledger lists archived paths and hashes.
The local recovery ZIP is intentionally excluded from Git. No scientific result
was regenerated or edited to strengthen a conclusion.
