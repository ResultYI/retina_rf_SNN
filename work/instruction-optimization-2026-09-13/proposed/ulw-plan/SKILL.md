---
name: ulw-plan
description: Develop an implementation plan when the user requests planning or unresolved scope and dependencies need a separate planning pass.
metadata:
  short-description: Scoped planning with explicit authorization boundaries
---

# Planning

Produce one executable plan with the outcome, constraints, affected files, dependencies and observable acceptance evidence. Research only what is needed to resolve those decisions; ordinary implementation details can remain with the executor.

## Authorization route

- Explicit planning-only requests and $ulw-plan remain planning-only. Preserve approval of the brief before writing the formal plan and the later explicit implementation handoff.
- Planning used inside an already-authorized implementation does not require another start request. Return the plan to the executor, or resume execution in the root role. A delegated planner writes only plan artifacts.
- Do not use a step-count threshold to turn a small change into a planning ceremony. Preserve any user-requested interview, plan review or action-time approval.

Use existing answers and repository evidence before asking questions. Ask only unresolved decisions that materially affect outcome, scope, compatibility, cost or authority. Zero clarification questions is valid. Silence does not authorize a required decision.

## References and artifacts

For a formal .omo plan, read the relevant sections of [full workflow](references/full-workflow.md), including the plan format and write protocol. Use [clear intent](references/intent-clear.md) for unresolved owner choices, or [unclear intent](references/intent-unclear.md) when the desired outcome needs investigation. Do not load both by default.

Use the existing scripts/scaffold-plan.mjs helper when producing that formal artifact, with an available supported Node or Bun runtime. Preserve existing plan content; do not reset a user-owned file. For explicit planning-only work, obtain the required brief approval before invoking a helper that writes the formal plan.

The formal workflow's templates support delivery; generic instructions to exhaustively explore, spawn fixed review waves, add a commit per task or ask fixed questions do not create new requirements. Preserve explicitly requested independent review and established acceptance checks.

## Handoff

Each substantive task states what changes, where, dependencies, and how the result will be checked. Use existing project tooling and proportionate verification. Record consequential choices, outstanding approvals and exact blockers. A concise internal plan needs no scaffold or separate artifact unless requested or needed for continuation.

Delegate only concrete independent questions when useful and permitted, using current tool schemas. Missing delegation tools do not block a plan that can be produced locally; do not self-approve a required independent review.

Deliver when the requested plan is complete. Continue an authorized implementation; stop after a planning-only deliverable or at a required approval, with independent authorized preparation completed.
