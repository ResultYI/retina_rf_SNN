# Schottdorf–Lee 22-cell prediction-baseline inventory

**Scope.** Read-only inventory on 2026-08-31. The reference data contract is the current shared-BC Canonical V1 artifact: 22 cells, 37 recordings, native 150 Hz bins, the 16/4 source-segment split, 30-bin warmup mask, Bernoulli event targets, and the stored inner-development contract (`80/20` within each training trial with a 60-bin guard). No model was loaded for inference or training.

## Evidence method and contract notation

- `Y` means the existing implementation/artifact explicitly establishes the stated item.
- `Y*` means the common raw-data loader, 22 cell IDs, split counts, source identity, and valid-bin counts match. The final shared-BC artifact does not persist a standalone target/mask tensor for a second bytewise artifact-to-artifact comparison.
- `N/A` means no fitted-model selection applies (the constant rate is estimated only from training bins).
- `No` means the requested baseline does not have that contract. `—` means no implementation/artifact exists.
- Reported NLL is the arithmetic mean of each cell's validation Bernoulli NLL. It is not pooled-bin weighted unless explicitly stated.

The current shared-BC results and the two existing baseline result bundles have identical 22 cell IDs and identical per-cell train/validation valid-bin counts (direct check: `shared_vs_ln_ids=True`, `shared_vs_final_ids=True`, `shared_vs_ln_bins=True`, `shared_vs_final_bins=True`). The current shared-BC training contract records the same `make_inner_dev` 80/20 plus 60-bin guard definition as the center-surround LN.

| Requested baseline | Existing state | Same 22 cells / split / loss mask / Bernoulli target | Same development-based model-selection protocol | Validation NLL and checkpoint/result evidence | Unique missing work |
|---|---|---|---|---|---|
| Constant rate | **已完成** | `Y* / Y* / Y* / Y*`; rate is estimated only from the training valid bins | `N/A` | **0.509817266**. The final-fair bundle stores per-cell constant NLL and explicitly checks training-only rate and the common data/target contract: `output/real_data/schottdorf_lee_2021_22cell_final_fair_prediction_benchmark_revision4/results.json`. The same value is independently present in `output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/results.json`. No model checkpoint exists or is needed. | None. |
| Center-surround separable LN | **已完成** | `Y* / Y* / Y* / Y*`; each of the 22 stored checks records `same_target_mask_identity=true` against the frozen 22-cell source | **Y** — uses `make_inner_dev`, 80/20 training-trial split, 60-bin guard, four fixed lambdas, inner-dev selection, then fresh full-train refit; original validation is excluded | **0.425997944**. Results/checks: `output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/results.json`; 22 `ln-trained.pt` checkpoints in its `cells/` subtree; protocol: `training/mechanistic_retina/center_surround_ln.py`, `evaluation/mechanistic_retina/schottdorf_center_surround_ln.py`. The stored provenance source is the older R4 result bundle, not a shared-BC checkpoint. | None for the data/prediction contract. Only an optional provenance re-anchor if one requires every final comparison artifact to name the shared-BC result bundle. |
| Spatial Contrast | **尚未实现** | `— / — / — / —` | `—` | No source implementation, output result, or checkpoint was found under `baselines/`, `training/`, `evaluation/`, `tests/`, `scripts/`, or `output/real_data/`. | Implement the frozen spatial-contrast term and run the 22-cell protocol. |
| Schottdorf–Lee original/adapted model | **尚未实现** | `— / — / — / —` | `—` | The repository contains the Schottdorf–Lee data adapter (`data/schottdorf_lee_2021.py`, `data/schottdorf_lee_multirecording.py`) but no original-model or adapted-model implementation, result bundle, or checkpoint. | Implement/adapt the specified original model, then fit it under the 22-cell contract. |
| Causal spatiotemporal CNN | **尚未实现** | `— / — / — / —` | `—` | No 2-D/spatiotemporal CNN implementation or checkpoint exists. The closest non-equivalent code is `baselines/graph_tcn.py`: spatial graph pooling followed by causal `Conv1d` blocks, not a spatial CNN. Its **old-R4-only** compact Graph-TCN run has NLL **0.474113636** and 22 checkpoints at `output/real_data/schottdorf_lee_2021_22cell_final_fair_prediction_benchmark_revision4/`; its selection uses training NLL only (`evaluation/mechanistic_retina/schottdorf_final_benchmark_run.py`, `evaluation/mechanistic_retina/schottdorf_neural_baseline.py`), so it does not satisfy the current development-selection contract. | Implement the requested causal spatiotemporal CNN and run it with the common inner-dev/refit protocol. |
| ConvGRU / CRNN | **尚未实现** | `— / — / — / —` | `—` | No ConvGRU, CRNN, GRU, or LSTM baseline class, result bundle, or checkpoint was found. | Implement one frozen ConvGRU/CRNN specification and run it with the common 22-cell contract. |

## Source-level evidence

- Constant/old compact-neural common-data assertions and the old-R4-only source guard: `evaluation/mechanistic_retina/schottdorf_final_benchmark_run.py:80-135`; per-cell data-contract verification: `evaluation/mechanistic_retina/schottdorf_final_benchmark_cell.py:42-107`.
- The current final shared-BC contract: `output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/results.json`; its manifest: `output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/run-manifest.json`.
- Center-surround LN Bernoulli objective, inner-dev selection, and fresh refit: `training/mechanistic_retina/center_surround_ln.py:108-187`; its 22-cell source loading/identity checks: `evaluation/mechanistic_retina/schottdorf_ln_source.py:64-108`; output/checkpoint writer and guard definition: `evaluation/mechanistic_retina/schottdorf_center_surround_ln.py:43-129`.
- Graph-TCN is explicitly a causal temporal `Conv1d` model: `baselines/graph_tcn.py:26-79`; its training-only selection is in `evaluation/mechanistic_retina/schottdorf_neural_baseline.py:76-157`.

## Recommendation

**First baseline to run: Spatial Contrast**, after implementation. It is the smallest missing comparison directly layered on the completed center-surround LN contract; do not treat the old Graph-TCN result as a substitute for the requested causal spatiotemporal CNN.
