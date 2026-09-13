---
name: study
description: Teach a topic, create a structured course or continue an existing learning program; not for isolated factual answers or code review.
---

# Study

Teach toward the learner's stated goal with reliable material, concrete explanations, examples and evidence of understanding. Use existing answers and learning preferences; ask only for information needed to choose a useful learning route.

A focused lesson or question can be taught directly in chat. It does not require creating a course, migrating unrelated state or opening a viewer. Do not infer mastery from merely viewing a lesson.

## Route by the requested work

| Request | Read when needed |
|---|---|
| Choose a new learning goal, scope, level or schedule | [Anchoring](references/phase-0-anchoring.md) |
| Research material for a new course | [Research](references/phase-1-research.md) |
| Generate course or module files | [Generation](references/phase-2-generation.md), then its format and language references |
| Continue a course, evaluate exercises or adjust pace | [Learning](references/phase-3-learning.md) |
| Review or consolidation | [Consolidation](references/phase-4-consolidation.md) and [scheduler](references/fsrs-scheduler.md) |
| Skill tree, XP or progress map | [Skill tree](references/skill-tree.md) |
| Read or write persisted learning state | [State schema](references/state-schema.md) |
| Old state must be used or updated | [Migration](references/migration-guide.md) before writes that depend on it |
| Start or operate a course viewer | [Viewer](references/learning-viewer.md) |

Load only the active workflow and its necessary dependencies. Do not reopen the full reference set for each response.

## New course

Establish the route, materials, baseline, time and storage path from the current request and existing answers. Preserve route confirmation, research-scope confirmation and the Module 00 review required by the course workflow. If the user has approved the route and explicitly asked to skip further confirmations, continue within that approved scope; silence does not grant that exception.

Use real sources before generation. Initialize state using the current schema without overwriting existing courses. The generation reference owns the course format: one section question per section page, a module preface and section pages, with scope sized to the learning goal. Length checks are advisory, not a substitute for content quality.

## Existing course

Continue from local course content and current metadata rather than restarting research. For full interactive lessons with course files, follow the viewer workflow; use chat for targeted questions, an explicit chat preference or unavailable viewer.

Read the learning record before grading or recording progress. Clear questions only after answering them, and update pace/depth feedback through the authorized state path. Add explanations and practice under 99-content-supplements/ after generation unless the user asks to revise original course content.

Check due reviews once at the first formal learning session of the day, with at most one compact notice. Local review dates do not create automations, notifications or wakeups.

## State and language

The state schema is the source of truth. Use scripts/write-state.py or its documented atomic write method; JSON write failure is not success. Use the existing migration and review helpers from this skill directory. Do not rebuild or overwrite state to hide an error.

Keep meta.json as the source for skill-tree and RPG settings, params.json for pacing/review preferences, and concepts.json for stored difficulty and stability; retrievability is computed. Preserve the current defaults and user choices.

In Chinese teaching, explain a term in Chinese and introduce its English name or abbreviation on first use. Preserve the requested depth and named exam points. Keep runtime caveats in the conversation instead of inserting them into learner-facing course content. Report progress only when supported by learning evidence.
