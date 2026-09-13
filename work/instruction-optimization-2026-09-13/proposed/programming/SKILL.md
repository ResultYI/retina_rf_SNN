---
name: programming
description: Implement or review Python, Rust, TypeScript or Go code when language-specific design, type safety or runtime guidance is needed.
---

# Programming

Use the project's existing architecture, supported dependencies and toolchain. A local edit does not authorize a migration, global installation, CI redesign or unrelated refactoring. Apply guidance only to the requested change and affected boundaries.

## Core constraints

- Preserve type safety and existing public contracts. Parse untrusted inputs at system boundaries, handle relevant errors, and make resource ownership and asynchronous cancellation explicit.
- Handle tagged variants exhaustively. Do not hide errors with Any, as any, unchecked assertions, ignored errors or type-check suppression. Keep unsafe operations inside documented, verifiable boundaries.
- Introduce a type, helper or dependency when it represents a real contract or simplifies the requested work. File size, parameter count or a library preference alone is not a correctness defect or permission to restructure.
- Preserve persisted formats, public behavior, scientific numerical contracts and explicit compatibility requirements. A shorter implementation is not equivalent evidence when operation order or backend identity is frozen.

## Read only relevant guidance

For a language-specific question, use the matching index and follow the links needed for that question:

| Language | Index |
|---|---|
| Python | [Python](references/python/README.md) |
| Rust | [Rust](references/rust/README.md) |
| TypeScript | [TypeScript](references/typescript/README.md) |
| Go | [Go](references/go/README.md) |

A small change whose contract is already clear does not require reopening an index. Linked library choices, project layouts, style rules and tool commands are recipes for matching tasks, not authority to replace a project's working stack. This scope rule governs the reference recipes; language safety requirements still apply.

For Rust unsafe, raw-pointer, FFI, MaybeUninit, custom lock-free or unsafe Send/Sync work, read [Rust UB guidance](references/rust-ub/README.md) and the safety references it requires before changing the boundary. Preserve required Miri and other safety checks; unavailable required evidence must be reported as INCONCLUSIVE.

## Verification

Use existing focused checks that exercise the requested behavior. For a substantive bug fix, reproduce the failure and obtain a meaningful regression check when feasible; do not add tests that merely copy implementation details. Run wider checks when shared impact or explicit acceptance requires them, not because an example lists them.

Do not weaken valid assertions or suppress errors to make checks pass. Missing tooling is an evidence gap, not a code defect. Report observed results and exact limitations; complete all required in-scope verification before claiming the implementation verified.
