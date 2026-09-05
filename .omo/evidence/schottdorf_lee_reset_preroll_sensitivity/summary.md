# Schottdorf–Lee reset/pre-roll sensitivity

STATUS: COMPLETED; no sensitivity pass/fail threshold.

Final shared-BC 22 trained checkpoints; 37 recordings. Original validation: 65,760 identical scored bins in all modes. No training, model edits, new probes, or checkpoint conversion.

## Exact evaluation contract

- Production: reset at each 150-bin (1 s) segment, score local bins 30–149; first 200 ms is warmup.
- 400 ms causal pre-roll: 60 actual preceding bins before each first scored bin, including the original 200 ms warmup. Input covers 180 bins; score only bins 60–179. This is not an additional 400 ms before the original segment.
- Continuous: one uninterrupted 3000-bin (20 s) forward from live t=0 for each recording/trial, without resets between its 1 s segments. Only original validation bins are scored. There is no state-transfer API change: original forward evaluates the complete causal prefix.
- Preceding stimulus and binary observed spikes are loaded from the original train/validation timeline. Observed history includes preceding unscored bins; the model's strictly-past shift remains unchanged. No cross-recording/trial or discontinuous context.
- H1-off, direct-BC-off and AC-off apply throughout each condition's entire context, not by splicing normal state into a clamped continuation.

Primary overall NLL is the unweighted mean across 22 cells, consistent with the prior 22-cell reporting. Also shown: bin-weighted NLL. Population |Δlogit| statistics and pathway magnitudes pool the 65,760 scored bins. ΔNLL and Δlogit are mode minus production. No pass thresholds are defined.

## Population

| Mode | Cell-mean NLL | ΔNLL | Bin-weighted NLL | ΔNLL weighted | Mean abs(Δlogit) | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| production | 0.438956146 | 0.000000000 | 0.437458485 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 |
| preroll_400ms | 0.438955046 | -0.000001100 | 0.437458388 | -0.000000097 | 0.000078522 | 0.000225544 | 0.018087268 |
| continuous | 0.438955113 | -0.000001032 | 0.437458468 | -0.000000017 | 0.000079869 | 0.000241184 | 0.018164992 |

## Pathway-off effect on the same scored bins

Each magnitude is mean |z_off − z_normal| within that mode, not the normal response's reset sensitivity.

| Mode | Pathway | Mean abs(Δlogit) | Change from production |
|---|---|---:|---:|
| production | H1_off | 0.033058075 | +0.000000000 |
| production | direct_BC_off | 1.384571142 | +0.000000000 |
| production | AC_off | 1.163596296 | +0.000000000 |
| preroll_400ms | H1_off | 0.033056253 | -0.000001822 |
| preroll_400ms | direct_BC_off | 1.384446821 | -0.000124321 |
| preroll_400ms | AC_off | 1.163513461 | -0.000082835 |
| continuous | H1_off | 0.033056194 | -0.000001881 |
| continuous | direct_BC_off | 1.384441663 | -0.000129480 |
| continuous | AC_off | 1.163510366 | -0.000085929 |

## Most sensitive cells

Ranked by largest absolute ΔNLL across the two alternatives; each cell listed once.

| Cell | Group | Mode | NLL | ΔNLL | Mean abs(Δlogit) | P95 | Max |
|---|---|---|---:|---:|---:|---:|---:|
| 67#34 | PC ON | preroll_400ms | 0.386399627 | 0.000018001 | 0.000369880 | 0.001810253 | 0.018087268 |
| 67#4 | PC OFF | continuous | 0.464882910 | -0.000013292 | 0.000083584 | 0.000253600 | 0.005346537 |
| 68#7 | PC ON | preroll_400ms | 0.385205179 | -0.000012547 | 0.000176228 | 0.000688142 | 0.013047457 |
| 68#10 | MC ON | preroll_400ms | 0.421058923 | -0.000012130 | 0.000078084 | 0.000271225 | 0.005682707 |
| 67#14 | PC ON | preroll_400ms | 0.372914523 | -0.000010222 | 0.000206276 | 0.000872302 | 0.013290167 |

## Verification and provenance

- 22 strict checkpoint loads; production logits bitwise equal to the saved validation logits and NLL exactly equal to the final recorded NLL.
- Original target/mask/source order exact; all modes score the same 65,760 bins and use the same targets/stimulus at those bins.
- Off contributions exact-zero; finite outputs; model state_dict unchanged and no parameter gradients.
- Current core source hashes differ from the original training manifest for the files listed below. No sources were changed here. Strict checkpoint load and full production replay were required, without converting checkpoints. Current hashes are recorded in verification.json; historical source-byte identity is not claimed.

- `D:\PythonProject\retina_rf_SNN\models\mechanistic_retina\bipolar_subunits.py`
- `D:\PythonProject\retina_rf_SNN\models\mechanistic_retina\canonical_contract.py`
- `D:\PythonProject\retina_rf_SNN\models\mechanistic_retina\model.py`
- `D:\PythonProject\retina_rf_SNN\models\mechanistic_retina\pathway_rf.py`
- `D:\PythonProject\retina_rf_SNN\models\mechanistic_retina\pathway_spatial_geometry.py`
- `D:\PythonProject\retina_rf_SNN\models\mechanistic_retina\shared_subunits.py`
- `D:\PythonProject\retina_rf_SNN\models\mechanistic_retina\spatial_contract.py`
- `D:\PythonProject\retina_rf_SNN\models\mechanistic_retina\support_partition.py`

## Artifacts

- per_cell.csv: 66 cell/mode rows with NLL, ΔNLL and mean/P95/max |Δlogit|.
- pathway_effects.csv: 198 cell/mode/pathway rows with absolute effect and change from production.
- population.csv and population_pathways.csv: pooled/equal-cell aggregate definitions above.
- evaluation_logits.pt: scored logits for all modes/clamps, targets and explicit recording/trial/live-bin identity mappings; not a model checkpoint.
- verification.json: contracts, checkpoint/source hashes and per-cell checks.
- inputs.py, run.py, report.py: evidence-only reproduction scripts.
