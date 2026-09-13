---
name: frontend
description: "MUST USE for frontend, web UI, UX, or visual work. Routes design references, verification of affected surfaces, and style lookups. Reuse project design conventions and tooling; apply Lighthouse targets and wider quality gates when requested or required by the established acceptance contract. Never weaken UX to improve a score."
---

# Frontend

This file is a router, not a rulebook. The rules live in three rulesets under `references/`; your first job is to load the smallest set of files that covers the request, state which you loaded in one sentence, then execute under their guidance. Loading nothing and freestyling produces the generic AI-slop output this skill exists to prevent; loading everything wastes context and creates contradictory instructions.

Apply every referenced workflow within the user's authorized scope and the project's established acceptance contract. A local UI change does not automatically require a new design-system document, dependency or skill installation, CI changes, deployment, or a whole-site audit. Use current session tool schemas; examples below are not permission to invent missing tools. Preserve explicit user/project release checks and approval requirements. Missing verification evidence is INCONCLUSIVE, not a demonstrated product defect: attempt bounded recovery, continue independent work, and report the exact unverified items without claiming PASS or waiving a required check.

## Phase 0 — Route (before any UI work)

| Request involves… | Read |
|---|---|
| ANY UI implementation, styling, redesign, mockup, or visual decision | `references/design/README.md` FIRST. Reuse existing tokens, components, and tooling, then route to relevant taste and brand references. |
| Writing or modifying frontend code, OR auditing performance / SEO / accessibility / quality | ALSO `references/perfection/README.md`. Verify affected behavior and applicable quality targets through representative browser evidence; do not drop features or hide content to improve scores. |
| Looking up a concrete style, color palette, font pairing, chart type, landing-page structure, or UX guideline — or generating a project design system from keywords | `references/ui-ux-db/README.md`. A searchable CSV database with a CLI; a lookup tool, not a posture. Load on demand; `design` stays the source of truth for taste and the `DESIGN.md` contract. |

**For implementation work, design + perfection load together.** Apply design fidelity and performance checks to the affected surface; neither visual polish nor a synthetic score alone proves the requested behavior works.

## Ruleset 1 — design (`references/design/`)

The reference library has one architecture file, 12 taste skills (Layer A — *how to execute*), and 69 brand design systems (Layer B — *what it should look like*). Most non-trivial tasks load **one Layer A + one Layer B**. `README.md` carries the full routing flow, stacking rules, anti-patterns, and the mandatory browser-based Design QA phase; `_INDEX.md` catalogs all 81 files with mood-to-brand mappings — read it whenever routing is not obvious from the tables below.

### Layer 0 — architecture

| File | Read when |
|---|---|
| `design-system-architecture.md` | Creating a design system is requested or materially needed for a new product or broad redesign; absence of `DESIGN.md` alone does not trigger it for a local edit. |

### Layer A — taste skills (pick AT MOST ONE style skill; they encode opposing philosophies)

| File | Read when the user says… |
|---|---|
| `taste-skill.md` | Nothing style-specific — "make a good UI". The default all-rounder. |
| `gpt-tasteskill.md` | "Awwwards-tier", "wow factor", "cinematic", "scroll-triggered", or `taste-skill` results felt too safe. |
| `minimalist-skill.md` | "minimal", "clean", "Notion-like", "Linear-like", "editorial". |
| `brutalist-skill.md` | "brutalist", "raw", "Swiss", "experimental", "anti-design". |
| `soft-skill.md` | "premium", "luxury", "calm", "expensive", "spa", "boutique", "elegant". |
| `redesign-skill.md` | Improving EXISTING UI — "this looks bad", "fix the design". Audit-first workflow; never use on greenfield. |
| `image-to-code-skill.md` | "Generate the design first, then code it." Pair with one imagegen file below. |
| `output-skill.md` | Stacks on any style skill when output is incomplete — placeholders, `// TODO`, half-done components. |
| `stitch-skill.md` | Stacks on any style skill for Google Stitch compatibility or a `DESIGN.md` doc export. |
| `imagegen-frontend-web.md` / `imagegen-frontend-mobile.md` / `imagegen-brandkit.md` | Image-only output (mockup, app-screen concepts, brand board). These NEVER write code — switch to `image-to-code-skill.md` if code is wanted. |

### Layer B — brand design systems (orthogonal to Layer A; stack freely)

When the user names a brand or site — "Linear-style", "like Stripe's landing" — load `references/design/<brand>.md` as the token source of truth (palette, type scale, components, do/don'ts). Coverage includes `apple` `stripe` `linear.app` `notion` `vercel` `claude` `figma` `airbnb` `nike` `tesla` `spotify` `raycast` `revolut` and ~56 more; the full list with mood shortcuts is in `_INDEX.md`. Extract the tokens and apply them to the project's own content — never copy logos or trademarked imagery. If the named brand is missing, fall back to a Layer A mood match or the `open-design` skill.

### React dev tooling

| File | Read when |
|---|---|
| `react-dev-tooling-skill.md` | React tooling setup is in scope, or an affected performance issue needs instrumentation beyond existing tools. Use its recipes only when the installation itself is authorized and needed. |

## Ruleset 2 — perfection (`references/perfection/`)

| File | Read when |
|---|---|
| `README.md` | Any frontend code is written or audited. Defines representative browser checks, applicable acceptance targets, root-cause diagnosis, UX preservation, and accurate evidence reporting. |
| `react-perf-tooling.md` | A React performance investigation requires render instrumentation. Use relevant recipes and established budgets; do not create new tooling or blanket failure thresholds for unrelated edits. |

When a Lighthouse audit is in scope, an available audit helper may be used (build for production first; do not present dev-server scores as production performance):

```bash
uv run $SKILL_DIR/scripts/perfection/lighthouse-audit.py https://localhost:3000
```

For a full performance audit, use the device profiles required by the acceptance contract and repeat noisy measurements as needed. Routine UI edits use focused browser and regression checks instead of automatically starting a full audit.

## Ruleset 3 — ui-ux-db (`references/ui-ux-db/`)

`README.md` documents the search CLI and the master-plus-overrides persistence pattern. The CLI (run from the ruleset directory so it finds `data/`):

```bash
python3 $SKILL_DIR/references/ui-ux-db/scripts/search.py "<query>" --design-system -p "Project"   # full design-system generation
python3 $SKILL_DIR/references/ui-ux-db/scripts/search.py "<query>" --domain <domain>             # targeted lookup
python3 $SKILL_DIR/references/ui-ux-db/scripts/search.py "<query>" --stack <stack>               # stack best practices
```

Domains: `product` `style` `typography` `color` `landing` `chart` `ux` `react` `web` `prompt`. Stacks: `html-tailwind` (default) `react` `nextjs` `vue` `svelte` `astro` `swiftui` `react-native` `flutter` `shadcn` `jetpack-compose`.

## Quick routes — most common requests

| Request | Load |
|---|---|
| "Build a landing page" (no direction given) | `design/README.md` + `design/taste-skill.md` + `perfection/README.md` |
| "Linear-style landing page" | `design/README.md` + `design/linear.app.md` + `design/taste-skill.md` + `perfection/README.md` |
| "Premium SaaS hero like Stripe" | `design/README.md` + `design/stripe.md` + `design/soft-skill.md` + `perfection/README.md` |
| "Improve this existing dashboard" | `design/README.md` + `design/redesign-skill.md` + `perfection/README.md` |
| "Audit my site" / "make this page faster" | `perfection/README.md` (+ `perfection/react-perf-tooling.md` when React render instrumentation is relevant) |
| "Mockup image of a fintech app" — no code | `design/imagegen-frontend-mobile.md` (+ a Layer B brand if named) |
| "What palette/fonts fit a wellness brand?" | `ui-ux-db/README.md` → search CLI |
| "Set up this React project" | `design/README.md` + `design/react-dev-tooling-skill.md` |

## Shared axioms (all three rulesets agree — apply always)

- **Reuse the project's design system.** Follow existing tokens and component conventions, whether documented in `DESIGN.md` or expressed in code; create or expand documentation when the requested scope needs it.
- **Never weaken UX to buy points.** No dropping animations, hiding content, or simplifying interactions for a score or a deadline.
- **No emojis as icons.** SVG icon sets only (Lucide, Heroicons, Radix, Phosphor).
- **GPU-composited animation only** — `transform`, `opacity`, `filter`; never animate layout properties.
- **Verify the affected UI in a real browser before claiming it verified.** Exercise relevant states and representative affected breakpoints. Missing browser access must be reported as unverified after bounded recovery, not silently skipped or treated as a product defect.

## When to load something else instead

| Situation | Load |
|---|---|
| Brand/style not among the 69 in `references/design/`, or the user says "Open Design" | `open-design` skill — the local nexu-io/open-design library (137+ design skills, 150+ design systems) |
| Driving a browser for the Design QA phase | An available browser skill/tool using its current schema; `agent-browser` if actually available and applicable |
| Pure TypeScript/logic work with zero visual surface | `programming` skill alone — this skill adds nothing there |

## Activation

Use for any frontend, web UI, UX, visual, design, styling, layout, animation, performance, accessibility, or SEO work — building, redesigning, auditing, or generating mockups. Not for backend, CLI, or pure-logic tasks with no visual surface.
