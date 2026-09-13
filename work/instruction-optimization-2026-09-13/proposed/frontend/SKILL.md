---
name: frontend
description: Build or revise web interfaces, layouts and interactions, or investigate frontend design, accessibility and performance.
---

# Frontend

Match the requested outcome and the project's existing tokens, components and tooling. A local UI edit does not require a new design system, broad redesign, dependency installation or whole-site audit.

## Route by the actual task

| Need | Reference |
|---|---|
| New design direction or substantial redesign | [Design index](references/design/README.md); follow the relevant style or brand link |
| Performance, accessibility, SEO or broad quality audit | [Quality guidance](references/perfection/README.md) |
| A specific palette, font, component or UX lookup | [UI/UX database](references/ui-ux-db/README.md) |
| Image-only web concept | [Web image workflow](references/design/imagegen-frontend-web.md) |
| Image-only mobile concept | [Mobile image workflow](references/design/imagegen-frontend-mobile.md) |

For a small change to an established component, use its existing design conventions directly. Load another reference only when it resolves a real design or implementation decision. Linked design and audit recipes apply within this scope; they do not impose a full design library or Lighthouse run on every edit.

Preserve the user's requested features and content. Do not hide UI, remove accessibility, or reduce interactions to improve a score. Keep image-only requests separate from requests for working controls.

## Verify the affected surface

Use available browser tooling to inspect changed states and relevant viewport sizes. Check interactions, loading/error states and accessibility affected by the change. Reuse valid evidence; broaden checks only for shared impact, failures or explicit acceptance requirements.

For a UI clone or design port, load visual-qa and satisfy its independent visual-comparison and code-fidelity checks. For other tasks, use visual-qa when structured visual comparison is needed. A screenshot proves appearance, not all functionality; a passing test or score does not prove appearance. If required browser evidence is unavailable after supported recovery, report INCONCLUSIVE and the exact gap, using a project-specific status label when required.

Preserve release and action-time approvals. This skill does not authorize publication, CI redesign, installing tools, or unrelated fixes. Backend or pure logic work without a UI consequence does not need this skill.
