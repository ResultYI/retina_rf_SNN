---
name: debugging
description: Diagnose and fix observed runtime failures, hangs, incorrect outputs or unexplained performance, including binary investigation.
---

# Debugging

Ground the diagnosis in observed behavior. A plausible explanation from source alone remains a hypothesis. An environment limitation or missing reviewer output is not automatically a product defect.

## Investigation

Reproduce the reported failure with the narrowest permitted scenario. Compare expected and actual behavior, identify plausible causes, and choose checks that distinguish them. Do not invent a minimum number of hypotheses or repeat unchanged checks. If evidence remains contradictory, reassess the approach and use an independent perspective when useful and permitted.

Read the matching runtime reference before attaching a debugger or relying on runtime-specific capture semantics:

| Runtime | Reference |
|---|---|
| Python | [Python](references/runtimes/python.md) |
| Node, Bun, Deno source | [Node](references/runtimes/node.md) |
| Rust | [Rust](references/runtimes/rust.md) |
| Go | [Go](references/runtimes/go.md) |
| Native binary | [Native binary](references/runtimes/native-binary.md) |
| Packaged JS or bundled application | [Bundled application](references/runtimes/bundled-js-binary.md) |

A familiar failure with adequate logs or an existing reproducer does not require a debugger attachment or every reference. Choose an available tool suited to the evidence; specialist tools are not universal prerequisites.

For a difficult investigation, load only the needed method: [environment and artifacts](references/methodology/00-setup.md), [hypothesis investigation](references/methodology/02-investigate.md), [cause and repair](references/methodology/06-fix.md), [runtime QA](references/methodology/08-qa.md), or [partial runtime evidence](references/methodology/partial-runtime-evidence.md). These procedures support the current task; their example phase counts, agent counts and tools do not impose extra work or authority.

## Repair and finish

Confirm the cause with evidence that distinguishes it from alternatives. Make the smallest authorized repair, then rerun the reported scenario and affected checks. Preserve failing-first regression evidence when practical and required; if the failure cannot be reproduced, state the limit instead of claiming a confirmed fix.

Track temporary probes and altered runtime state so they can be removed or restored safely. Preserve useful evidence, user artifacts and concurrent edits. Do not globally roll back after a failed attempt.

Read-only or diagnosis-only requests do not authorize repairs. Debugging does not authorize commits, installations, production changes or external actions. Preserve user/project approval, data-access and verification boundaries. Report an evidenced defect as FAIL and missing required evidence as INCONCLUSIVE.
