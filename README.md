# Retina RF SNN

This repository trains a causal retinal spiking model on ISETBio cone-response sequences. The active research path is intentionally singular:

```text
ISETBio cone response
-> H1 local surround
-> ON/OFF sustained/transient bipolar channels
-> local recurrent amacrine state
-> one unlabeled heterogeneous RGC pool
-> encoder-transpose tied decoder
-> clean current cone-contrast reconstruction
```

Each cone center owns two anonymous RGC units by default. Spatial scale, kinetic mixture, membrane and adaptation dynamics, inhibition, threshold, and subunit adaptation are learned per unit. Functional cell-type names are assigned only after training when preregistered cluster relationships are satisfied.

The decoder reuses the encoder's dense masked-softmax spatial weights. Its only learned parameters are bounded positive per-unit ON/OFF gains and one bias per cone. It has no independent receptive field or temporal filter.

## Experiment

The sole experiment configuration is [`configs/experiment.yaml`](configs/experiment.yaml), loaded through strict nested dataclasses. Unknown keys are rejected.

```powershell
python scripts/run_experiment.py `
  --config configs/experiment.yaml `
  --device cuda
```

The training contract uses 320 time steps: 64 no-gradient burn-in steps followed by one state detach and a 256-step differentiable chain. The last 96 steps are supervised. Activation checkpointing divides the differentiable region into 32-step blocks without truncating state gradients.

The objective combines current reconstruction, an augmented-Lagrangian spike-energy inequality, wiring cost, unit variance floor, same-center phenotype repulsion, and rate homeostasis. All components live in [`loss/retina.py`](loss/retina.py).

Dynamic receptive fields use matched low/high contexts from the same source clip and an identical final probe. The trained model selects one shared unit plan that is reused by the saved initialization reference. Evidence is aggregated within source before a paired source-level bootstrap comparison. Jacobians and finite differences use the same continuous spike-probability readout, and temporal RGC typing uses only effective radius plus impulse, step, and flicker responses. Dynamic RF and typing run only after reconstruction and the fixed target-energy gate pass.

## Checkpoints

The active schema is:

```json
{"schema": "retina_rf_snn", "schema_revision": 3}
```

Revision 3 checkpoints preserve independent source-sampling and augmentation RNG state, validation state, the frozen reference and target energy budgets, optimizer, and scheduler. Revision 2 is intentionally incompatible. Training also writes one `initial_reference.pt` beside the checkpoints so final evidence can compare trained behavior with the exact deterministic initialization. A checkpoint becomes energy-feasible only after the budget ramp ends, and final evaluation uses hard validation energy divided by the fixed target budget.

## Static and runtime checks

The refactor that established this structure was code-only. Run checks locally before training:

```powershell
python -m pytest -q
```

ISETBio HDF5 generation utilities remain separate from the experiment runner. Existing HDF5 exports, run artifacts, and checkpoints are not rewritten by the source refactor.
