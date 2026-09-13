---
name: review-work
description: Review a completed implementation against its requirements, changed behavior and affected risks, with evidence-backed findings.
---

# Implementation review

Review the actual requested change. Establish the goal, constraints, relevant revision or file set, and available verification evidence. Do not assume HEAD~1 or main is the correct comparison base; preserve concurrent work and use an isolated worktree if another checkout is necessary.

## Select the relevant perspectives

| Perspective | Evidence to inspect |
|---|---|
| Goal and constraints | Requested behavior, omissions, scope and compatibility |
| Correctness | Changed logic, relevant callers, error paths and regression checks |
| Runtime QA | The affected user scenario and observable result |
| Security | Changed trust boundaries, permissions, data exposure or dependencies |
| Context | Specific unresolved requirements or prior decisions relevant to this change |

Cover applicable perspectives with adequate evidence. No fixed agent count, minimum scenario count or whole-repository context search is required. Start with the request, source and existing checks; search history or connected services only for a concrete relevant gap.

Use independent reviewers when required by the user/project or when material risk warrants them and delegation is permitted. Specify the exact scope, evidence and deliverable; use current tool schemas and keep ownership separate. The implementer cannot replace a required independent reviewer. A requested comprehensive review retains all of its specified perspectives and independence.

## Evidence and outcome

Exercise relevant runtime behavior when the review requires runtime evidence. Source inspection cannot establish that a live scenario passed. For text-only changes, inspect the affected document and its meaningful constraints instead of inventing application scenarios.

- PASS: all applicable required checks have supporting evidence and no blocking defect remains.
- FAIL: a concrete defect is supported by evidence.
- INCONCLUSIVE: required evidence or independence is missing.

A timeout, silent reviewer or unavailable tool is not a defect or a pass. Try a bounded supported alternative and preserve completed results. Report exact remaining gaps; do not waive an explicit gate, start a debugging loop without a defect, or invent product changes.

Lead with actionable findings, ordered by severity and supported by file locations, observed behavior and practical impact. State the scope and verification limits. With no findings, say so without claiming untested behavior verified. Keep secrets and private data out of reports.

A review-only request ends with the report and does not authorize fixes. An implementation request includes correction of its own defects within scope. Review completion does not grant publication, merge, installation or external messaging authority, and does not replace required user acceptance.
