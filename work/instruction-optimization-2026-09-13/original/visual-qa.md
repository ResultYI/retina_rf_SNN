---
name: visual-qa
description: "Rigorous visual QA for any UI you built or changed, across BOTH web/page UIs and TUI/terminal UIs. MUST USE after building or changing any UI to verify it visually before declaring it done. Captures objective reference evidence with a bundled diff script (image-diff for screenshots, tui-check for terminal captures), then runs two parallel read-only oracle passes (design-system and functional integrity; visual fidelity and CJK precision) and synthesizes one good/bad verdict. Triggers: visual QA, visual regression, screenshot diff, pixel diff, image comparison, UI looks wrong, design system check, is this really a design system or just an image, alpha channel breakage, responsive check, CJK text, Korean/Japanese/Chinese text clipping, baseline drop, glyph drop, TUI alignment, terminal UI, tmux capture, box-drawing border misalignment, wide-character column drift. Use it even when the user does not say visual QA but asks whether a page, component, or terminal layout looks right."
---

## Codex Harness Tool Compatibility

Examples below describe roles and workflow, not guaranteed tool names or schemas. Use the current session's available spawn, message, wait, browser, terminal, and image tools with their documented parameters. Select a named reviewer only if that role is available; otherwise give an available independent reviewer the same self-contained charter, evidence, files, and constraints. Do not pass stale `fork_context`, model, or role parameters that the live schema does not support.

A wait timeout alone does not prove failure. When evidence or a reviewer is unavailable, try a bounded supported alternative and report any remaining gap as INCONCLUSIVE. This permits an accurate status report, not a passing verdict. The implementer cannot self-approve a required independent review. Fallback does not authorize global tool installation, CI changes, publication, or bypass of explicit user/project safety and release approvals.

# Visual QA - Dual-Oracle Web and TUI Verification

Verify the affected rendered UI against intent using objective evidence plus two parallel read-only review passes, then synthesize GOOD, NEEDS WORK, or INCONCLUSIVE. The script numbers focus the reviewers. They are not the verdict. Keep the review within the changed behavior and its affected boundaries; a scoped UI edit does not require inventing a site-wide design system or repairing unrelated styles.

## Purpose and when to use

- Use after you build or change any UI, before calling it done. Covers web/page UIs and TUI/terminal UIs.
- Use when output must match a mock, a baseline, or a stated design intent; when you suspect a regression; when CJK (Korean/Japanese/Chinese) text may clip, misalign, or wrap awkwardly; when a claimed design system might actually be a flat image; when a terminal layout may overflow or its borders may break.
- Skip when there is no rendered surface (pure backend or library logic with no visual or terminal output). For broad post-implementation review use review-work; this skill is the visual specialist.

In the commands below, `$SKILL_DIR` is this skill's own directory (the folder containing this SKILL.md). The bundled script lives at `scripts/cli.ts` inside it.

## Step 1 - Detect the surface

- Web/page UI: renders in a browser (HTML/CSS/JS, components, canvas, SVG). Evidence is screenshots.
- TUI/terminal UI: renders as text in a terminal (box-drawing, panes, status lines, REPL/TUI apps). Evidence is terminal captures.

If the change touches both, run both capture tracks and feed both into the passes.

## Step 2 - Capture objective reference evidence

### Web

1. Capture a REFERENCE image when the user supplied a mock/target or a known-good baseline is available. Otherwise use the stated intent and existing design conventions; do not invent a reference or block a simple change solely because no baseline image exists.
2. Capture the ACTUAL affected surface using existing project or session browser tooling. Match the reference viewport when comparing images. If a named tool is unavailable, use an available equivalent. Do not automatically install a global browser tool; missing capture capability requires bounded recovery and an exact unverified-gap report.
3. When a reference and the diff tool are available, run the diff and keep the JSON. Otherwise record the available direct visual evidence and the comparison limitation:

```
bun "$SKILL_DIR/scripts/cli.ts" image-diff <reference.png> <actual.png>
```

Key fields: `dimensionsMatch`, `diffRatio` (0..1), `similarityScore` (0..100), `alphaChannelIntact`, `hotspots[]` (grid regions ranked by `diffRatio`).

### TUI

1. Capture plain text and an ANSI-preserving copy:

```
tmux capture-pane -p > capture.txt
tmux capture-pane -e -p > capture-ansi.txt
```

2. Run the check with the REAL terminal width and keep the JSON:

```
bun "$SKILL_DIR/scripts/cli.ts" tui-check capture.txt --cols <N>
```

Key fields: `maxWidth`, `overflowLines[]`, `borderMisaligned`, `wideCharColumns[]`, `hasAnsi`.

This JSON (diff ratio, similarity score, hotspots or overflow lines, border alignment, wide-char columns, alpha) is REFERENCE evidence to aim the reviewers. It is not the verdict by itself.

## Step 3 - Dispatch two read-only QA subagents in parallel

Send both review requests concurrently when supported. Each reviewer is read-only and returns PASS, REVISE, FAIL, or INCONCLUSIVE with concrete evidence or exact missing evidence. Pass A checks the requested implementation and existing design conventions; Pass B directly opens captures and inspects source/content for visual and CJK defects. Keep both passes scoped to affected surfaces and preserve any explicitly required independent review.

Paste evidence directly into each prompt: source code, the plain-text TUI captures, the script JSON, and the screenshot paths plus your described observations for web. The two passes differ in depth by charter, not by any model or effort setting, which cannot be pinned per call.

### Pass A - Design-system and functional integrity (deeper, strict)

```
task(subagent_type="oracle",
  run_in_background=true,
  load_skills=[],
  description="Visual QA pass A: design-system and functional integrity",
  prompt="""
REVIEW TYPE: DESIGN-SYSTEM AND FUNCTIONAL INTEGRITY (read-only)
TIER INTENT: Treat this as the deeper, stricter pass. Reason exhaustively before concluding. Assume a plausible-looking surface may be faked or mock-only until the source proves otherwise.

INTENT:
{What the user asked for, the mock or baseline, and the constraints.}

SURFACE: {web | tui | both}

SOURCE CODE:
{Full source of the UI: components, styles/tokens, layout, render code. Include neighboring files that show existing patterns.}

CAPTURES:
{Web: actual screenshot path(s) plus your described observations. TUI: paste capture.txt and capture-ansi.txt inline.}

SHARED SCRIPT EVIDENCE (reference, not verdict):
{Paste the image-diff or tui-check JSON. Use alphaChannelIntact for the transparency check.}

CHECK EACH:
1. Implementation and design conventions: does the changed surface follow the existing tokens/primitives and satisfy the user's requested interactions? A pasted image standing in for requested live controls is a defect. Lack of a site-wide reusable system is not by itself blocking for a scoped UI edit or requested static artifact; require a new design system only when it is part of the user/project acceptance contract.
2. Faked-with-an-image anti-pattern: is the UI a real DOM/component tree, or a pasted raster/screenshot or background-image standing in for live elements? For TUI: a real layout that reflows, or hardcoded pre-rendered text at fixed widths?
3. Alpha and transparency: handled correctly, with no unexpected opaque or black fills and correct PNG/CSS alpha? Cross-check alphaChannelIntact.
4. Code style and implementation quality.
5. Responsive and resize behavior across viewport sizes (web) or terminal resize (TUI).
6. Do the user-intended FEATURES actually work: interactions, states, navigation (web); input handling, resize, scroll (TUI)? Trace the code paths.

OUTPUT:
VERDICT: PASS | REVISE | FAIL | INCONCLUSIVE
CONFIDENCE: HIGH | MEDIUM | LOW
SUMMARY: 1-3 sentences
FINDINGS: for each, [dimension] [severity] what is wrong, where (file/line or capture region), and the concrete fix
WHAT IS GOOD: correct aspects that must not regress
BLOCKING: items that must be fixed; empty if PASS
"""
)
```

### Pass B - Visual fidelity and CJK precision (focused)

```
task(subagent_type="oracle",
  run_in_background=true,
  load_skills=[],
  description="Visual QA pass B: visual fidelity and CJK precision",
  prompt="""
REVIEW TYPE: VISUAL FIDELITY AND CJK PRECISION (read-only)
TIER INTENT: Treat this as the focused visual pass. Directly open the screenshots with the available image-viewing tool (`view_image`, `look_at`, or browser inspection) before judging. Anchor every claim to the script evidence, source code, and captures.

INTENT:
{What the user requested and the mock or baseline to match.}

SURFACE: {web | tui | both}

CAPTURES:
{Web: actual and reference screenshot paths plus your described observations. TUI: paste capture.txt and capture-ansi.txt inline.}

SOURCE CODE:
{For web: include the rendered text/content, components, typography, layout, and style code. For TUI: include render code that controls wrapping, width, and wide-character handling.}

SCRIPT EVIDENCE (consume available fields; state missing comparisons):
{Paste available image-diff or tui-check JSON, or the direct capture evidence and exact tool/reference limitation.}

USE THE EVIDENCE:
- Web (image-diff): start from diffRatio and similarityScore, then directly open every screenshot path and inspect every hotspots[] entry (gridX, gridY, x, y, width, height, diffRatio). Explain the visual cause of each flagged region from the pixels and source/content together.
- TUI (tui-check): inspect maxWidth vs expectedColumns, every overflowLines[] entry, borderMisaligned, and wideCharColumns[].

CHECK:
1. Does the rendered output match what the user requested: layout, spacing, color, type, alignment?
2. CJK precision:
   - Web: natural CJK line breaking for display and body text. Flag oversized headings that create orphaned one-character or final-syllable lines, split Korean/Japanese/Chinese semantic phrases unnaturally, detach labels such as `[Image #1]` from their content, clip baselines/descenders, drop glyphs (tofu), or show font metric mismatch. Treat the screenshot pattern `에이전트 오케스트 / 레이션 현황 및 미 / 래` as REVISE/FAIL, not acceptable wrapping.
   - TUI: wide-character column drift (CJK cells counted as 1 instead of 2), box-drawing border misalignment, content overflowing past the terminal width.

OUTPUT:
VERDICT: PASS | REVISE | FAIL | INCONCLUSIVE
CONFIDENCE: HIGH | MEDIUM | LOW
SUMMARY: 1-3 sentences
EVIDENCE TRACE: each hotspot or overflow line mapped to its visual cause
FINDINGS: for each, [severity] what is wrong, where (hotspot grid or capture line:col), and the concrete fix
BLOCKING: items that must be fixed; empty if PASS
"""
)
```

## Step 4 - Synthesize one verdict

When both passes return, merge them into a single report. Per dimension, mark good, bad, or unverified with evidence. For each bad item, state what is wrong, where (file/line, hotspot grid, or capture line), and the concrete fix. Missing evidence is unverified, not a product defect. Preserve correct behavior.

Completion gate: do not declare the affected UI verified and complete until both required passes are satisfied, OR the remaining gaps are explicitly listed and accepted by the user where that acceptance is permitted by the governing requirements. A high `similarityScore` with a broken requested feature is still a FAIL. After bounded supported recovery, an INCONCLUSIVE report is allowed and must state exactly what remains unverified; it does not waive independent review, approve leftovers, or satisfy a release gate.

```markdown
# Visual QA - Verdict: GOOD | NEEDS WORK | INCONCLUSIVE

| Dimension | Pass | Verdict | Evidence |
|---|---|---|---|
| Design system real vs faked | A | good/bad | ... |
| Features work | A | good/bad | ... |
| Responsive / resize | A | good/bad | ... |
| Alpha / transparency | A+B | good/bad | ... |
| Visual fidelity to intent | B | good/bad | ... |
| CJK precision | B | good/bad | ... |

## Must fix
[Blocking items, each with location and fix, in priority order]

## Good, keep it
[Correct aspects that must not regress]

## Completion gate
[Satisfied, or the exact remaining gaps and who accepted them]
```

## Step 5 - Clone-coding mode (when the task was a clone or design port)

Run this step IN ADDITION to Steps 1-4, but ONLY when the original user task was a clone or design port: "clone this site", "move this Figma design to code", "rebuild this screen", "make it look exactly like X". For these tasks run the following two additional verifications against the requested fidelity criteria. Repair concrete mismatches within scope; missing evidence follows the INCONCLUSIVE recovery/reporting rule above.

1. Pixel-perfect design-compare subagent (visual oracle). Dispatch a focused, read-only design-compare reviewer (recommend `gpt-5.5` with medium reasoning). It must crop/zoom BOTH the reference (the target / Figma export / source-site screenshot) and the ACTUAL screenshot into matching regions and read them **pixel-by-pixel** - header, nav, each card, spacing, type ramp, color tokens - not at a glance. Anchor every claim with the bundled tool:

```
bun "$SKILL_DIR/scripts/cli.ts" image-diff <reference.png> <actual.png>
```

   It judges whether layout geometry, spacing, design tokens (color, type, radius, shadow), and the design itself are identical to the target, region by region. Anything off by more than rounding is a finding.

2. Code-level design-system fidelity (code oracle). Use `lazycodex-clone-fidelity-reviewer` when available, or another independent reviewer with the same charter. It verifies a real component tree, design tokens and reused primitives appropriate to the requested clone, and no pasted image standing in for requested live UI. A named-role fallback does not waive independence; if no independent reviewer can run, report that required review as INCONCLUSIVE.

RULE: a clone is not verified until both the visual comparison and independent code review confirm the requested layer, design-system, and visual fidelity criteria on the same revision. Repair located mismatches and rerun the affected checks; broaden re-verification when a change affects prior evidence. Repeated failed attempts trigger diagnosis or an independent perspective, not blind full-loop retries. If a required comparison remains unavailable after bounded supported remedies, report INCONCLUSIVE and the exact gap. Do not claim a complete clone from one evidence track or self-approve a required independent review; preserve explicit user acceptance and release requirements.

## Reference evidence is not the verdict

The script quantifies pixels and columns. It cannot judge whether the result is a real design system, whether features work, or whether intent was met. A 99/100 `similarityScore` can still hide a pasted-image fake, a broken interaction, or clipped CJK descenders. Use the numbers to aim the oracles, then trust the synthesized review.

Illustrative output (locked field names):

```json
{
  "command": "image-diff",
  "dimensionsMatch": true,
  "reference": { "width": 1440, "height": 900 },
  "actual": { "width": 1440, "height": 900 },
  "totalPixels": 1296000,
  "diffPixels": 38880,
  "diffRatio": 0.03,
  "similarityScore": 97,
  "alphaChannelIntact": true,
  "hotspots": [
    { "gridX": 2, "gridY": 0, "x": 960, "y": 0, "width": 480, "height": 300, "diffRatio": 0.21 }
  ],
  "summary": "97/100 similarity; one hotspot in the top-right header region."
}
```

```json
{
  "command": "tui-check",
  "expectedColumns": 80,
  "lineCount": 24,
  "lineWidths": [80, 80, 82, 80],
  "maxWidth": 82,
  "overflowLines": [ { "line": 3, "width": 82 } ],
  "borderMisaligned": true,
  "wideCharColumns": [12, 13],
  "hasAnsi": false,
  "summary": "Line 3 overflows 80 cols by 2; borders misaligned at wide-char columns 12-13."
}
```
