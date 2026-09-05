# Retina RF — Canonical V1

This project fits real RGC responses with a physiology-constrained recurrent/state-space retinal point-process model. Macaque is the primary biological target. The historical repository name contains SNN; the current implementation is not a strict LIF/threshold-reset SNN.

**Independent audit: start at [AUDIT_INDEX.md](AUDIT_INDEX.md).** It identifies final results, code, checkpoints, limitations and historical dependencies. [CURRENT_STATE.md](CURRENT_STATE.md) is a navigation summary, not independent validation.

```text
visual stimulus -> species/experiment-specific cone front-end -> cone representation
 -> H1 -> shared ON/OFF sustained/transient BC encoder
          | narrow BC support -> direct BC --------------------> RGC
          | broader overlapping support -> BC -> AC dynamics --> RGC
 -> Bernoulli spike likelihood
```

Both BC views share encoder parameters. AC receives the broad BC representation and has no independent stimulus encoder. H1 amplitude, BC weights, AC group mixtures, per-cell BC/AC gains, bounded tau and explicit fractional delays are learnable. See [source](models/mechanistic_retina/model.py) and [mathematical contract](docs/architecture.md).

## Frozen evidence

Final lineage: `output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/`. It contains 22 cell-wise fits from 37 MC/PC recordings, native 150 Hz, measured Bernoulli spike events and 17x17 calibrated L+M Weber input. MC maps to parasol and PC to midget. No cross-recording population geometry is inferred. This is luminance fitting, not a full chromatic cone front-end.

| Model | 22-cell mean validation NLL |
|---|---:|
| Constant | 0.509817266 |
| Center-surround LN | 0.425997944 |
| Compact causal CNN | 0.422597811 |
| Canonical V1 | 0.438956146 |
| SC-adapted (supplement) | 0.458313940 |

These are existing artifact values, not new fits. NLL is nats per valid bin, equally averaged over cells. The [prediction package](.omo/evidence/final_prediction_results/README.md) contains per-cell values and paired uncertainty. Canonical has 33 trainable/optimizer-listed scalars per cell (129 registered parameter scalars), CNN 2990; this is not matched-capacity evidence.

The [illusion aggregation check](.omo/evidence/parametric_illusion_benchmark/aggregation_check/summary.md) qualifies earlier cohort-reversal claims: Mach width>0 reversal disappears after balancing ON/OFF or the four classes. Control-subtracted AC interactions and class-resolved curves, including null results, are retained.

## Runtime and reproduction

The recorded final local runtime is Python 3.12.7, PyTorch 2.6.0 CPU and NumPy 2.2.6. `requirements.txt` is the historical dependency list, not a fully pinned environment; inspect each artifact's manifest.

The final training producer is `run.py` inside the final lineage directory and uses `training/mechanistic_retina/r4_development.py`. The generic `scripts/run_schottdorf_multirecording_training.py` belongs to an earlier fixed-step protocol and is not the final experiment entry point. Inspection and aggregation do not require training.

See [data availability](audit/DATA_AVAILABILITY.md) for raw data, local-only tensors and unresolved frame-zero synchronization. Small final checkpoints and selected frozen analysis tensors are included in the intended Git publication set.

## Cleanup and scope

The 2026-09-05 cleanup archived superseded unreferenced experiments and old reports with verified hashes. Model, training, loader, baseline and test Python sources remain unchanged. Some legacy modules/results remain because of imports, CLI defaults or artifact provenance; their presence does not make them current evidence.

See the [cleanup record](audit/cleanup_20260905/README.md) and [Pro audit prompt](audit/PRO_AUDIT_PROMPT_ZH.md).
