---
name: visual-qa
description: Inspect rendered web or terminal interfaces for visual regressions, reference fidelity, responsive layout and text clipping.
---

# Visual QA

Inspect the affected rendered surface against the user's request and existing design conventions. Capture actual output with available browser or terminal tools. Check changed states, relevant viewport or terminal sizes, and affected interactions. Do not invent a reference image when none was provided.

For an ordinary UI edit, direct inspection and focused interaction checks may suffice. No fixed reviewer count is required. Preserve any explicit independent-review or release requirement.

## Comparison tools

Use the bundled scripts/cli.ts helper when a comparison is useful and its runtime is available:

    bun "<skill-dir>/scripts/cli.ts" image-diff <reference.png> <actual.png>
    bun "<skill-dir>/scripts/cli.ts" tui-check <capture.txt> --cols <actual-width>

Match viewports for image comparison. Capture terminal output with the actual width and ANSI information when relevant; tmux is an option, not a prerequisite. Do not install a global capture tool solely to follow an example.

The image diff reports dimensionsMatch, diffRatio, similarityScore, alphaChannelIntact and hotspots. The terminal check reports overflowLines, borderMisaligned and wideCharColumns. Use these to locate issues; numeric similarity alone is not a passing verdict.

Inspect layout, spacing, type, contrast and transparency as affected. For CJK text, look for missing glyphs, clipped baselines, unnatural line breaks and wide-character column drift. Check that requested controls function and resize correctly.

## Clone or design port

When the request requires matching a source design, compare corresponding regions of the actual and reference output at matching dimensions. Also inspect the component and token implementation: a pasted image cannot stand in for requested live controls.

For every clone/design-port task, obtain independent visual comparison and independent code-level fidelity review. A reviewer who did not implement the change may cover both tracks unless the user/project requires separate reviewers. Use current available roles and schemas; no model is hardcoded. Both tracks must cover the same revision. A missing track is INCONCLUSIVE, not a completed clone; use a project-specific status label when required.

## Report

Report located defects and supporting captures or source locations. GOOD requires the requested visual and functional checks; NEEDS WORK requires an observed mismatch; INCONCLUSIVE identifies missing evidence. Recheck affected evidence after fixes, not every unchanged surface.

A review-only request does not authorize repairs. Do not treat missing tools as a product defect or waive required independent review, user acceptance or publication approval.
