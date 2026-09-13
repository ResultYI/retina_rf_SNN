# Retina Research Execution Rules

Codex executes the requested repository task. The user and ChatGPT own scientific interpretation, research planning and subsequent experiments.

* Stay within the requested scope; do not choose research directions, claims or next experiments. Stop when complete.
* Preserve frozen Canonical V1. Changing architecture, loss, training protocol, parameter bounds, data splits or evaluation criteria requires explicit instruction. Every implementation change needs a scientific or correctness reason stated in the task. Never change the model or protocol merely to improve NLL, accuracy, recovery scores or appearance.
* Distinguish code facts, experimental results, model-internal inference and biological conclusions. Model outputs are model results, not biological claims. Use established retinal neuroscience, computational neuroscience and machine-learning terminology.
* Change only necessary files. Do not refactor, rename, reformat, clean or silently repair unrelated work. Add no unnecessary baselines, audits, robustness tests, ablations or engineering; run only tests and experiments required to verify the task.
* Read-only tasks permit no file modifications. No-training tasks permit neither training nor new training checkpoints. Insufficient evidence must be reported as `UNVERIFIED`.
* Report observations concisely; interpret scientific meaning only when asked. Do not narrate already-visible tables/figures or add promotional language, exaggerated conclusions or filler.
