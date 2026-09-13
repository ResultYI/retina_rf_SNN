---
name: ponytail
description: Find a simpler implementation when the user requests minimal code, fewer dependencies, YAGNI or an overengineering check.
argument-hint: "[lite|full|ultra]"
license: MIT
---

# Ponytail

Satisfy the complete requested outcome with the simplest correct implementation. Reuse project code, the standard library, native platform features and installed dependencies before adding new machinery.

Apply to the relevant task. If the user explicitly requests a continuing mode, retain it until changed; implicit selection does not create an every-response mode.

## Choices

Inspect the changed path and callers needed to understand behavior and impact. Do not scan every caller or the whole repository without a concrete dependency question.

Avoid speculative abstractions, configuration, scaffolding and dependencies. Prefer clear local code over a clever one-liner. Existing persistence, compatibility, security, accessibility and scientific contracts are requirements, not optional complexity.

Do not ship a reduced feature and ask whether the user still wants the original. If a simpler substitute changes an explicit requirement, present the concrete tradeoff and obtain a decision before substituting it. Review findings do not authorize cleanup.

## Intensity

- lite: implement the requested behavior; mention a materially useful simpler alternative.
- full: prefer the simplest adequate design within the agreed requirements.
- ultra: question speculative additions more aggressively, while preserving every explicit requirement.

The default is full. No level grants permission to remove requested features or skip required evidence.

## Verification and output

Use the project's existing focused checks. Add a meaningful regression check when substantive changed logic needs one; neither a mandatory new demo nor a new testing framework is required for each edit. Never weaken valid tests.

Document a deliberate shortcut only when its real limit affects maintenance or the user's decision; keep any useful ponytail: marker accurate. Explain the result and material limitations concisely. Provide requested reports or walkthroughs in full.
