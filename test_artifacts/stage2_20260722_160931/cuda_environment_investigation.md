# CUDA environment investigation

Goal: locate the existing CUDA-capable Python environment and resume the bounded stage-two smoke without installing dependencies.

## Hypotheses

1. **H1 — alternate Conda environment:** an existing Conda environment contains a CUDA-enabled PyTorch build. Distinguishing evidence: enumerate Conda environments and execute each environment's Python to print `torch.__version__`, `torch.version.cuda`, and `torch.cuda.is_available()`.
2. **H2 — project/IDE interpreter:** the project or IDE points to a local/non-Conda Python executable different from `D:\anaconda\python.exe`. Distinguishing evidence: inspect project interpreter metadata and installed Python paths, then execute each candidate directly.
3. **H3 — import-path mismatch:** the active interpreter has CUDA packages available but imports a CPU-only `torch` from another location. Distinguishing evidence: record `torch.__file__`, `sys.path`, and Conda package metadata for the active environment.

## Initial observations

- `nvidia-smi`: NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB total, 6173 MiB free.
- `D:\anaconda\python.exe`: PyTorch `2.6.0+cpu`, `torch.version.cuda == None`, `torch.cuda.is_available() == False`.

## Candidate interpreter results

| Environment | Python | PyTorch | CUDA runtime | Available |
|---|---:|---:|---:|---:|
| base | 3.12.7 | 2.6.0+cpu | None | false |
| SeqTrack | 3.10.16 | 1.11.0 | 11.3 | true |
| llm_algo | 3.10.20 | 2.12.0+cpu | None | false |
| retina_rf_snn | 3.11.15 | 2.5.1 | None | false |
| seqtrack | 3.10.16 | 1.11.0 | 11.3 | true |
| snn_env | 3.11.15 | 2.10.0+cu126 | 12.6 | true |

H1 confirmed: `D:\anaconda\envs\snn_env\python.exe` is the current CUDA-capable project candidate. H3 is refuted for the base interpreter because it imports its own CPU build from `D:\anaconda\Lib\site-packages\torch`.

## Selected runtime

- Interpreter: `D:\anaconda\envs\snn_env\python.exe`
- PyTorch: `2.10.0+cu126`
- CUDA runtime: `12.6`
- CUDA tensor allocation: passed on `cuda:0`
- Project imports: NumPy 1.26.4 and h5py 3.16.0 available in `snn_env`.
- PyYAML: absent from `snn_env`; `D:\anaconda\Lib\site-packages\yaml` loads in pure-Python mode when appended after the environment site-packages. PyTorch remains loaded from `snn_env`.
- Older SeqTrack CUDA environments were rejected: PyTorch 1.11 lacks the current checkpoint/LR scheduler API contract.

## Source instrumentation applied

- `scripts/run_experiment.py`: added only `gradient_norm`, `temporal_gradient_norm`, and `peak_memory_bytes` to the validation row, as explicitly allowed by the stage-two instructions.

## Recovered smoke outcome

- Artifact directory: `test_artifacts/stage2_20260722_161857/`
- CUDA smoke: exit code 0, 10 optimizer steps, 5 validation records.
- Checkpoint resume: exit code 0, restored optimizer step 10, no duplicate training rows.
- Final status: `STAGE_2_PASS`; `stage_3_budget_smoke = GO`.

## Artifacts

- This file is retained as stage-two diagnostic evidence.
- The recovered CUDA smoke is retained in `test_artifacts/stage2_20260722_161857/`; the earlier blocked directory was not overwritten.
- No debugger ports, temporary source edits, environment changes, or background processes are planned.
