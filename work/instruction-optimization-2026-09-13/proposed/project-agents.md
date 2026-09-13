# Retina Research Execution Rules

Execute the requested task through its required verification, then stop. Scientific interpretation, research planning and subsequent experiments remain with the user and ChatGPT unless explicitly requested.

- Preserve frozen Canonical V1. Changes to architecture, loss, training protocol, parameter bounds, data splits or evaluation criteria require explicit instruction. Every implementation change needs a scientific or correctness reason stated in the task. Never alter the model or protocol merely to improve NLL, accuracy, recovery scores or appearance.
- Separate code facts, experimental results, model-internal inference and biological conclusions. Model outputs do not establish biological mechanisms. Use established retinal neuroscience, computational neuroscience and machine-learning terminology; report insufficient evidence as UNVERIFIED. If a tool or skill calls missing evidence INCONCLUSIVE, report it as UNVERIFIED here and identify the gap.
- Read-only tasks permit no file modifications. No-training tasks permit neither training nor new training checkpoints. Honor any restriction on reading new spike targets.
- Change only necessary files and preserve concurrent work. Do not add unrelated refactoring, cleanup, baselines, audits, robustness tests, ablations or engineering.
- Read the files and references needed for the current task. Use existing project tooling and only the checks needed to verify the requested change. Verification must stay within the task's data-access, training and frozen-protocol boundaries.
- Continue authorized implementation and required verification without another start request. Ask only for missing user-only information, a material scope decision or a required approval; pause only dependent work.
- Report observations and verification limits concisely. Interpret scientific meaning only when asked; do not repeat visible tables or figures. Completion requires the requested deliverable and applicable evidence, with any unresolved requirement identified explicitly.
