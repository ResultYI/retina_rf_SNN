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

Dynamic receptive fields use matched low/high contexts from the same source clip and an identical final probe. Jacobians and finite differences use the same continuous spike-probability readout. A finite-difference comparison is marked `threshold_crossing_not_local` only when the perturbation changes the target unit's hard events. Dynamic RF and temporal cell-type probes run only after reconstruction and energy gates pass.

## Checkpoints

The active schema is:

```json
{"schema": "retina_rf_snn", "schema_revision": 2}
```

Revision 2 checkpoints preserve independent source-sampling and augmentation RNG state, validation state, energy state, optimizer, and scheduler. Training writes `checkpoint_last.pt`, `checkpoint_best_reconstruction.pt`, and, once the energy constraint is active, `checkpoint_best_feasible.pt`. Final evaluation selects the best feasible checkpoint first.

## Static and runtime checks

The refactor that established this structure was code-only. Run checks locally before training:

```powershell
python -m pytest -q
```

ISETBio HDF5 generation utilities remain separate from the experiment runner. Existing HDF5 exports, run artifacts, and checkpoints are not rewritten by the source refactor.
