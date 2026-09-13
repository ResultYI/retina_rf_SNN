---
name: programming
description: "MUST USE for ANY work on .py .pyi .rs .ts .tsx .mts .cts .go files. One philosophy: strict types, modern stacks (Pydantic v2 / serde+thiserror / Zod / gin+sqlc+pgx+slog), modern toolchains (uv+basedpyright+ruff / cargo+clippy+miri / Bun+Biome+tsc / gofumpt+golangci-lint v2+nilaway+go-race), parse-don't-validate, exhaustive match, typed errors, no any/unwrap/panic, scoped architectural review and behavior-focused verification. Routes to references/{python,rust,typescript,rust-ub,go}/. Triggers: write/edit Python/Rust/TypeScript/Go code, new project, gin server, bubbletea TUI, CJK IME, connect-go RPC, sqlc pgx, branded ids, exhaustive match, unsafe Rust, miri, oversized file, refactor, TDD, e2e test, arena, allocator, bumpalo, const fn, const generics, comptime, zero-alloc, bitfield, repr, scopeguard, errdefer, Zig-like, zerocopy, packed struct."
---

# Programming

You are a senior engineer who writes Python, Rust, and TypeScript with one shared discipline. **Type-strict. Stack-first. Async-correct. Architecturally honest about file size.**

This skill is an index. The hard per-language rules live under `references/`. Load the language-specific reference **before** writing a single line of code.

## Scope and verification contract

Apply this skill and its references to the requested change and affected boundaries. Explicit user/project safety, compatibility, testing, and release requirements remain binding. Existing toolchains take precedence over the defaults below; a local edit does not authorize a toolchain migration, global installation, CI redesign, branch-wide cleanup, or unrelated restructuring. Use tools and parameters actually available in the current session. If a named tool is unavailable, use an existing equivalent and state any material evidence gap; never fabricate a passing result.

Choose the smallest effective verification for the changed behavior. Reuse relevant tests and add focused regression or characterization evidence where needed. A reversible, low-impact edit does not automatically require a new test or every test layer. Run broader suites when the affected boundary, a failure, or an explicit acceptance contract requires them. The test and size rules in this section govern the scope of recipes in linked references; language safety requirements remain in force. After bounded supported recovery, missing required evidence may be reported as INCONCLUSIVE with exact gaps, but cannot be called passed or used to waive required independent review or approval.

---

## PHASE 0 — LANGUAGE GATE (RUN THIS FIRST, EVERY TIME)

**DO NOT WRITE OR EDIT A SINGLE LINE OF CODE BEFORE COMPLETING THIS GATE.**

1. **Identify the language** from the file extension or the user's request.
2. **STOP** and read the matching reference set:

   | File / Language | MANDATORY reading (load `Read` tool on every file below) |
   |---|---|
   | `.py`, `.pyi`, "Python" | `references/python/README.md` + every file under `references/python/` that the README tells you to load on demand |
   | `.rs`, `Cargo.toml`, "Rust" | `references/rust/README.md` + every file under `references/rust/` that the README tells you to load on demand. **IF the change touches `unsafe`, `*mut`, `*const`, `MaybeUninit`, FFI, `unsafe impl Send/Sync`, or a custom lock-free primitive: ALSO load `references/rust-ub/README.md` plus every file under `references/rust-ub/`.** |
   | `.ts`, `.tsx`, `.mts`, `.cts`, "TypeScript" | `references/typescript/README.md` + every file under `references/typescript/` that the README tells you to load on demand |
   | `.go`, `go.mod`, `go.sum`, `.golangci.yml`, `*.proto` next to a Go module, "Go" / "Golang" | `references/go/README.md` + every file under `references/go/` that the README tells you to load on demand |

3. Only after the references are loaded, apply the **shared philosophy** below plus the per-language iron list from the reference.

Apply the relevant language safety rules to small and one-off code too. Size the tooling and verification to the task using the scope contract above; do not bootstrap or migrate infrastructure solely to perform a disposable local operation.

---

## Shared philosophy (all three languages)

These are not style preferences. They are the six axioms every recipe in `references/` derives from.

1. **The type system is your proof system.** Make illegal states unrepresentable. The compiler / type checker is the cheapest test you will ever run. If a bug can be expressed as a type error, it is *required* to be expressed as a type error.

2. **Parse, don't validate.** Untrusted input crosses a boundary exactly once - at the boundary it is parsed into a typed value (Pydantic v2 in Python, `serde` + `#[derive]` in Rust, Zod in TypeScript). Inside the boundary, code receives typed values and never re-validates. The boundary owns trust; the interior owns logic.

3. **One name = one concept.** A `UserId` is not a `string`. A `Seconds` is not a `Milliseconds`. Use `NewType` (Python), newtype tuple structs (Rust), or branded types (TypeScript) for every distinct semantic primitive. The compiler refuses to let two semantic units mix.

4. **Exhaustive variant matching, always.** Discriminated unions and enums are matched exhaustively. Python: `match` + `case unreachable: assert_never(unreachable)`. Rust: `match` (the compiler enforces). TypeScript: `switch` + `assertNever`. **`if`/`elif`/`else` is forbidden for discriminating on a tagged variant** - it silently swallows new variants.

5. **Trust framework guarantees. Validate only at boundaries.** No null checks for values the type system already proves non-null. No `try/except` around code that cannot raise. No `unwrap`/`!`/`as` to paper over a contract you should have encoded in types. No defensive layer for a scenario you cannot name.

6. **Behavior-focused verification.** Use evidence that distinguishes the requested behavior from a regression. Prefer failing-first regression tests for substantive behavior fixes and use existing checks or direct inspection for low-impact changes when sufficient. See the verification discipline below.

---

## Verification discipline

For substantive behavior changes that need new regression coverage, use the red → green → refactor loop below. Reuse existing tests when they already prove the behavior. Do not create tests that merely mirror an implementation or require new infrastructure for a trivial reversible edit. Explicit project or user test requirements still apply.

### The order

1. **Red.** Write a failing test that names the behavior in `Given / When / Then`. Run it. *Confirm it fails for the right reason* — not a typo, not an import error. A test that fails because the function does not exist yet is the right reason. A test that fails because of a missing import is not.
2. **Green.** Write the minimum code to make the test pass. Resist adding the second case until the first passes. The second case is the next red.
3. **Refactor when needed.** With the test green, make only restructuring required by the change. Preserve valid test assertions and behavior contracts; never weaken a failing test to make the implementation pass.

### The shape of the test pyramid

Choose the layers that exercise the changed behavior and affected boundary; this table is not a requirement to add all three to every feature:

| Rung | Count | Purpose | Speed budget |
|---|---|---|---|
| **Unit** | many | Pure-function correctness for every meaningful input class (happy + edges + boundaries + error paths) | < 10 ms each |
| **Integration** | some | The real adapter against the real downstream (DB, queue, HTTP) — via `testcontainers`, `httptest`, or equivalent. NEVER a unit test pretending to be integration. | < 1 s each |
| **E2E scenario** | few | One narrative per user-visible outcome. Spins the binary or the full app; drives it through its real surface (HTTP route, CLI invocation, TUI keystroke). Asserts the *observable outcome*, not internal state. | seconds, run on CI |

For a user-facing behavior change, exercise its real surface when feasible and required by the task. This can be a focused manual scenario or an existing automated check; add E2E infrastructure only when the scope or acceptance contract requires it. If required surface evidence cannot be obtained, report the exact gap without claiming that behavior verified.

### Given / When / Then is mandatory

Every test — unit, integration, E2E — is structured by these three blocks. Names follow `Test_<Behavior>_when_<Condition>` or the language idiom (`it("<does X> when <Y>")`, `#[test] fn behavior_when_condition`).

```
Given: the preconditions and fixtures
When:  the single action under test
Then:  the observable outcome AND only that outcome
```

One `When` per test. Multiple `When`s = multiple tests. The `Then` asserts only what changed because of the `When` — not unrelated invariants.

### Less mock, the better

Mocks are a last resort, not a default. The priority order:

1. **Real object.** Use it when constructable in <1 ms (most domain types, pure functions, value objects).
2. **In-memory fake.** A real implementation of the interface backed by a map/slice — for stores, caches, queues. The fake has its OWN test that proves it behaves like the real one.
3. **Testcontainer / sandbox.** Real Postgres, real Redis, real S3-compatible (MinIO), via `testcontainers`. Slow but truthful.
4. **HTTP-level fake.** `httptest.Server` (Go), `respx` (Python), `msw` (TS) — fake at the wire, not at the SDK.
5. **Mock.** Only when 1–4 are genuinely infeasible (clock, randomness, external SaaS with no sandbox). Then mock the **narrowest** seam — never an entire service. A mock that returns whatever the test wants is a tautology and proves nothing.

**The rule**: if your test fails when the production code's *implementation* changes but its *behavior* did not, the test is over-mocked. Delete the mock; assert on observable outputs.

### Efficient AND accurate — both, not either

- **Accurate**: the test fails for the bug it names, and only that bug. No incidental coupling to format, ordering, whitespace, or unrelated fields. Assert on the *contract*, not on the dump.
- **Efficient**: the whole unit suite runs in < 30 seconds on a developer laptop. The whole integration suite in < 5 minutes. If you cross those budgets, profile and split — fast tests run on every save, slow ones run on push.
- **Deterministic**: no `sleep`, no wall-clock dependence, no order dependence (`-shuffle=on`, pytest-randomly, vitest random seed). Inject a `Clock`. Subscribe to the event, do not poll for it. Time-based flake is a bug, not a test issue.
- **Isolated**: every test starts from a known fixture and tears down. `t.TempDir()`, `t.Setenv()`, transactional rollback for DB tests. Two tests passing individually but failing together is a fixture leak — fix it immediately.

### Prompt tests follow the same rule

When tests cover LLM prompts or agent outputs, assert on **parsed structure, decisions, or rule data**, never on exact prompt strings. Pinning a sentence is brittle pretend-coverage; asserting that the prompt instructs the model to refuse on category X is real coverage.

### Anti-patterns the skill rejects

| Anti-pattern | Why it fails | Fix |
|---|---|---|
| Deferring required regression evidence until after delivery | Leaves changed behavior unverified. | Obtain the relevant evidence before claiming verified; prefer red first when adding regression coverage. |
| One mega-test asserting 12 things | First failure hides the next 11. | Split by `Then` clause — one assertion class per test. |
| Mocking every collaborator | Test passes regardless of real behavior. | Use a fake or the real thing. Mock only true unmockables. |
| `time.sleep(0.1)` to "let it finish" | Flake guaranteed. | Subscribe to the completion signal; bounded await. |
| Snapshot tests for everything | Locks formatting, not behavior. | Snapshots for *structure* (CLI help, JSON shape). Assertions for *behavior*. |
| Removing a failing test to "unblock CI" | You just deleted a bug report. | Fix the code or fix the test — never delete to silence. |
| `assert result is not None` and stopping there | Passes when result is garbage. | Assert the *value*, not its existence. |
| Ignoring affected error paths | Regressions often appear at boundaries. | Cover relevant edges with the narrowest useful check; use E2E when the whole interaction matters. |

---

## Cross-language iron list

Apply the language safety rules together with the scope and verification contract above.

| Rule | Python | Rust | TypeScript | Go |
|---|---|---|---|---|
| Immutable by default | `@dataclass(frozen=True, slots=True)` / Pydantic `frozen=True` | every binding is `let` (not `let mut`) unless mutation is the documented purpose | every field is `readonly`; arrays are `readonly T[]` | value types, unexported fields, no mutation methods unless mutation is the purpose |
| Branded primitives | `UserId = NewType("UserId", int)` | `struct UserId(u64);` (newtype tuple) | `type UserId = Brand<string, "UserId">` | `type UserID string` + smart constructor with unexported field |
| Exhaustive variant matching | `match` + `assert_never` | `match` (compiler-enforced) | `switch` + `assertNever` | sealed interface + type switch + **`exhaustive` linter** (the compiler will not help) |
| No untyped escape hatches | no `Any` in public sigs, no `cast`, no `# type: ignore` | no `unwrap`/`expect` outside `main`/tests, no `as` for narrowing, no `#[allow]` to silence real warnings | no `any`, no `as` (except `as const`, `satisfies`), no `!`, no `@ts-ignore`, no `@ts-expect-error` | no `interface{}` / bare `any` in domain sigs; no `_ = err`; no `//nolint` without reason |
| No bare error strings | typed exception dataclass with `__str__` | `thiserror` enum (lib) or `anyhow` with `.context(...)` (app) | `Error` subclass with typed fields | sentinel `errors.New` + typed `*XError` struct; wrap with `%w`; check via `errors.Is/As` |
| Boundary catch only | catch the exact exception you expect; broad `except Exception` only in `main()`, with logging + re-raise | `?` everywhere; never `panic!` in library code | `catch` must narrow with `instanceof` and re-throw or convert; no empty catch | every `(T, error)` checked; `panic` only in `main`/tests; one `httperr.Write` funnel in handlers |
| Resources via RAII | `with` (sync) / `async with` (async) | `Drop` impl or RAII guard | `using`/`await using` (TC39 explicit resource management) | `defer x.Close()` immediately after acquisition; `bodyclose`/`sqlclosecheck` linters enforce |
| Async runtime is mandatory | `anyio` (NEVER bare `asyncio`) | `tokio` (`async-std` is unmaintained) | platform-native async (Bun/Node) with structured cancellation via `AbortSignal` | `context.Context` as first param + `errgroup` for structured concurrency; `-race` on every test |
| Modern HTTP client | [`httpx2`](https://github.com/pydantic/httpx2) with HTTP/2 + brotli + zstd | `reqwest` with rustls | `ky` (default) / `undici` direct API (Node perf) - NEVER bare `fetch` in prod | stdlib `net/http.Client` with tuned `Transport` + `go-retryablehttp` for retry/backoff |
| No parameter mutation | params are inputs; produce a new value | `&mut` only when mutation is the documented purpose | parameters never reassigned (`noParameterAssign`) | value receivers when not mutating; pointer receivers only for genuine mutation; `copylocks` vet enforces |
| No helpers for one-off | inline a 3-line operation; do not abstract until the second caller | same | same | same |

---

## Modern ecosystem - canonical libraries (2026)

Use these unless the project's manifest explicitly picks something else.

| Domain | Python | Rust | TypeScript | Go |
|---|---|---|---|---|
| Data validation / boundary parse | **Pydantic v2** | **serde** + `#[derive(Deserialize)]` + `validator` | **Zod v4** (Standard Schema) | `validator/v10` (HTTP) + `protovalidate` (proto) + smart constructors (domain) |
| Internal value object | `@dataclass(frozen=True, slots=True)` | newtype tuple struct or plain `struct` | `type` alias with `readonly` | struct with unexported fields + `NewX(...)` constructor |
| Error types | typed exception dataclass | `thiserror` (lib) + `anyhow` (app) | `Error` subclass + Result pattern | sentinel `errors.New` + typed `*XError` struct + `%w` wrap |
| HTTP client | [`httpx2`](https://github.com/pydantic/httpx2) | `reqwest` | `ky` / `undici` | stdlib `net/http` + `go-retryablehttp` |
| Web framework | **FastAPI** | **axum** | **Hono** + `hono-openapi` | **gin** (de facto, ~48%) / `chi` (minimalist) / `connect-go` (RPC) |
| ORM / DB | SQLAlchemy 2.x async + `asyncpg` | `sqlx` (compile-time checked) | **Drizzle** | **sqlc** (codegen from `.sql`) + `pgx/v5` + `goose` migrations |
| CLI | **typer** + `rich` | **clap** (derive) + `color-eyre` + `indicatif` | `@clack/prompts` + `commander` | **cobra** + `huh` (prompts) + `slog` |
| Logging / observability | `structlog` (prod) or `rich.logging` (dev) | **tracing** + `tracing-subscriber` | `pino` (structured JSON) | stdlib **`log/slog`** (NEVER logrus/zap/zerolog for new code) |
| Testing | `pytest` | `cargo nextest` + `proptest` + `insta` | `bun test` / `vitest` | stdlib `testing` + `testify/require` + `goleak` + `autogold` + `rapid` + `testcontainers` |
| Data / analytics | **polars** + **duckdb** + `numpy` (NEVER pandas) | `polars-rs` or `arrow` | (defer to backend service) | `arrow-go` + DuckDB-Go bindings + `gonum` |
| LLM / agent | **pydantic-ai** | (call out to Python via subprocess) | **Vercel AI SDK** | direct `net/http` + Connect (langchaingo not recommended) |
| TUI | **textual** | `ratatui` | `@clack/prompts` or ink | **bubbletea v2 RC** + `bubbles/v2` + `lipgloss/v2` (v2 mandatory for CJK IME) |
| Config from env | **pydantic-settings** | `figment` or `config` | `zod` + `process.env` | `caarlos0/env/v11` (struct-tag env) |

A bare default constructor for any of these (no timeouts, no pool tuning, no schema) is a bug. See the per-language reference for the canonical production defaults.

---

## Modern toolchain - defaults for new setup

Use the project's existing supported toolchain for existing work. These defaults do not independently authorize installation or migration and do not turn every listed whole-project command into a mandatory check for a scoped edit.

| Tool category | Python | Rust | TypeScript | Go |
|---|---|---|---|---|
| Package / project manager | **uv** (NEVER pip/poetry/conda) | **cargo** + `cargo-nextest` + `cargo-machete` + `cargo-deny` | **Bun** (runtime + package manager); pnpm if Node is forced | **`go modules`** + `go work` for monorepos |
| Type checker | **basedpyright** with `typeCheckingMode = "all"` | the Rust compiler with `-D warnings` + clippy `pedantic` + `nursery` + `cargo` groups | `tsc --noEmit` (or `tsgo` when available) with `strict` + `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes` + `verbatimModuleSyntax` | the Go compiler + **`golangci-lint v2`** with the strict bundle + **`nilaway`** (nil-deref static analysis) |
| Linter + formatter | **ruff** with `select = ["ALL"]` | `clippy` + `rustfmt` | **Biome** (single binary - replaces ESLint + Prettier) | **`gofumpt`** (stricter gofmt) + `goimports -local` + `golangci-lint v2` |
| Test runner | **pytest** | **cargo-nextest** | `bun test` / `vitest` | stdlib `go test -race -shuffle=on -count=1` + `goleak` |
| UB / soundness gate | (n/a) | **nightly miri** with strict provenance + Tree Borrows pass | (n/a) | **`nilaway`** + `-race` detector + `goleak` are the equivalent gate |
| Disposable scripts | **PEP 723** inline metadata + `uv run script.py` | **rust-script** with inline `Cargo.toml` block | `bun run script.ts` | `//go:build ignore` + `go run script.go` |
| Bootstrap a new project | `scripts/python/new-project.py` | `scripts/rust/new-project.py` | `scripts/typescript/new-project.ts` | `scripts/go/new-project.py` |
| Pre-commit / CI gate | `ruff check . && basedpyright && pytest` | `cargo +nightly clippy -- -D warnings && cargo nextest run && cargo +nightly miri test` | `bunx biome check . && bunx tsc --noEmit && bun test` | `gofumpt -l . && golangci-lint run ./... && nilaway ./... && go test -race -shuffle=on -count=1 ./...` |

A `tsconfig.json` with `"strict": true` alone is **not** strict. The reference enumerates the additional flags. Same for `pyproject.toml` and `Cargo.toml` - the references contain the canonical full configuration.

---

## CODE SMELLS — AUTOMATIC REVIEW TRIGGERS

The smells below are prompts to inspect the changed code and its affected boundaries. A file exceeding 250 pure LOC or an unrelated pre-existing smell is not by itself a defect or a prerequisite refactor. Restructure only when necessary for the requested behavior or an explicit project rule; otherwise note a material concern briefly and keep the change scoped. These scope rules also apply when consulting `references/code-smells.md`.

Full rationale, measurement methods, workaround detection, and split examples: **[`references/code-smells.md`](references/code-smells.md)**.

### Smell 1 — File exceeds 250 pure LOC

A source file past 250 non-blank, non-comment lines warrants a responsibility check when relevant to the change. Length alone does not prove that splitting is useful. Measure: `awk '!/^[[:space:]]*$/ && !/^[[:space:]]*(\/\/|#|--)/' <file> | wc -l`.

**When relevant:** Identify the file's responsibilities and whether the requested change needs a boundary adjustment. Load `/refactor` if a structural change is needed. Do not require a split or an exemption comment solely to cross a line-count threshold.

### Smell 2 — Function with more than 3 parameters

More than 3 arguments signals the function is doing too much, or that related parameters belong in a typed struct. **Workarounds count as the same smell** — passing `dict`/`Record<string, unknown>`/`map[string]any`/`**kwargs`/`...args` to smuggle parameters through one argument, or a throwaway "config" object with 6+ fields that exists solely to wrap what would otherwise be positional args (genuine reusable domain types like `HttpClientConfig` are fine).

**When detected:** Group related parameters into a typed value object with a domain name. If 4+ independent inputs are genuinely required, the justification must be SPECIFIC. See [`references/code-smells.md` Smell 2](references/code-smells.md#smell-2--function-with-more-than-3-parameters) for examples in every language.

### Smell 3 — Redundant verification after a destructive action

In production code, repeating an assertion already guaranteed by an operation can be redundant. This does not apply to task-level verification of an external effect, uncertain tool results, destructive-action safeguards, or user/project-required evidence.

**When relevant:** Remove a redundant production check only when its contract is established and the removal is within scope. Preserve checks that address a real failure mode or a required safety/evidence contract. See [`references/code-smells.md` Smell 3](references/code-smells.md#smell-3--redundant-verification-after-a-destructive-action) for examples subject to this boundary.

### Smell 4 — Negative-form names and conditions

Naming variables, functions, or flags by the **absence** of a quality (`isNotValid`, `noErrors`, `cannotProceed`, `DisableLogging`) instead of its **presence** (`isValid`, `isClean`, `canProceed`, `LoggingEnabled`). Every negation forces the reader to invert mentally; two negations (`if !isNotReady`) become a logic puzzle nobody reviews confidently.

**When detected:** Rename to the positive form and invert the branch logic. Negation IS appropriate in guard clauses (`if !authorized { return }`) and filters (`items.filter(|x| !x.is_expired())`) — the negative form is the intent there. See [`references/code-smells.md` Smell 4](references/code-smells.md#smell-4--negative-form-names-and-conditions) for the full naming table and examples.

---

## MANDATORY POST-WRITE REVIEW LOOP

Review the changed code before claiming the task complete. Scale the review to its behavior and affected boundaries; existing unrelated smells are not automatic repair work.

### Step 1 — measure

For every file you created or modified:

```bash
awk '!/^[[:space:]]*$/ && !/^[[:space:]]*(\/\/|#|--)/' <file> | wc -l
```

Or run the per-language checker the skill ships:

```bash
# Python
uv run scripts/python/check-no-excuse-rules.py <changed paths>
# Rust
bash scripts/rust/check-no-excuse-rules.sh <changed paths>
# TypeScript
bun run scripts/typescript/check-no-excuse-rules.ts <changed paths>
```

### Step 2 — interpret

| Pure LOC | Verdict | Required action |
|---|---|---|
| ≤ 200 | Healthy | continue |
| 200 - 250 | Review cue | Inspect responsibilities if relevant to this change. |
| > 250 | Review cue | Assess the touched unit; split only when required by the change or an explicit project rule. |

### Step 3 — architectural self-review (always, even at 80 LOC)

Assess these questions for the changed code. Report material findings and verification results, without reciting the entire checklist:

1. **Single responsibility?** Does the changed unit have a clear responsibility, and is a structural change necessary to fulfill the task safely?
2. **Boundary purity?** Did I parse untrusted input into a typed value at the boundary, or did I pass `dict[str, Any]` / `serde_json::Value` / `unknown` past the boundary? If the latter, fix it.
3. **Variant discrimination?** Did I use `if`/`elif`/`else` (or `switch` without `assertNever`, or `match` without `assert_never`) anywhere to discriminate on a tagged type or enum? If yes, rewrite as exhaustive match.
4. **Escape hatches?** Any `Any`, `# type: ignore`, `unwrap`, `expect` outside `main`/tests, `as` numeric cast, `!`, `@ts-ignore`, `@ts-expect-error`, `#[allow]` on a real warning? If yes, fix the type or document why with a comment.
5. **Defensive layer?** Any null check, try/except, or `isinstance` guarding a value the type system already proves? If yes, delete.
6. **Helpers for one-off?** Any function, class, or trait introduced for a single caller that will never get a second caller? If yes, inline.
7. **Verification?** Does the selected evidence cover the changed behavior and affected boundary under the scope contract above?
8. **Parameter bloat?** Any function I wrote or modified that takes more than 3 parameters — or smuggles them through a dict/kwargs/`...args`/throwaway options object? If yes, group related params into a typed value object. See [Smell 2](references/code-smells.md#smell-2--function-with-more-than-3-parameters).
9. **Redundant verification?** Does production code repeat a guaranteed contract without a real purpose? Remove only scoped redundancy; retain external-effect verification and explicit safety/evidence checks. See [Smell 3](references/code-smells.md#smell-3--redundant-verification-after-a-destructive-action).
10. **Negative naming?** Any variable, function, or flag named by the absence of a quality (`isNotValid`, `noErrors`, `DisableX`) when a positive name (`isValid`, `isClean`, `EnableX`) would work? If yes, rename to positive form and invert the branch. See [Smell 4](references/code-smells.md#smell-4--negative-form-names-and-conditions).

Fix defects introduced by the change and unmet requirements within the authorized scope before claiming complete. Report material pre-existing issues separately; their discovery does not authorize unrelated repair or prevent a scoped task from completing. Preserve all explicit safety and acceptance requirements.

### Step 4 — if you need to refactor right now, invoke the right skill

- The user requested restructuring, or a structural change is necessary to fulfill the authorized task: **load the `refactor` skill** and apply its safe-refactor protocol to that scope.
- The user requested branch cleanup: **load the `remove-ai-slops` skill** for that authorized scope. Merely inheriting a branch with AI-generated patterns does not authorize branch-wide cleanup.

Use these skills when their scoped work is needed; a count of smells or lines alone does not require invoking them.

---

## Companion skills - explicit invocation triggers

| Trigger | Skill to load | Why |
|---|---|---|
| The user requests restructuring, or it is necessary to fulfill the authorized task | `refactor` | Safe refactoring within the requested boundary, with relevant behavior verification. |
| The user requests "remove slop", "clean AI code", "deslop", or equivalent branch cleanup | `remove-ai-slops` | Behavior-preserving cleanup within the authorized scope. |
| Rust code touches `unsafe`, `*mut`, `*const`, `MaybeUninit`, FFI, `unsafe impl Send/Sync`, or a custom lock-free primitive | `references/rust-ub/` | Full UB taxonomy + Miri strictness escalation. Every `unsafe` block must survive Miri Level 3 (strict provenance + symbolic alignment + preemption) before it ships. |

---

## Per-language jump table

**Stop. Read the matching reference fully before writing code.**

### Python (`.py`, `.pyi`)

**READ `references/python/README.md` FIRST.** Then load on demand:

| Need | Load |
|---|---|
| Strict pyproject.toml / basedpyright / ruff config | `references/python/pyproject-strict.md` |
| Type patterns (`NewType`, `Final`, `TypeGuard`, `Protocol`) | `references/python/type-patterns.md` |
| Data modeling (Pydantic vs dataclass vs TypedDict vs StrEnum) | `references/python/data-modeling.md` |
| Error handling (typed exceptions, exhaustive match, union returns) | `references/python/error-handling.md` |
| Async with anyio (task groups, cancel scopes, channels) | `references/python/async-anyio.md` |
| httpx2 production defaults (HTTP/2, brotli+zstd, pool tuning) | `references/python/httpx2-optimization.md` |
| **orjson** in hot paths (FastAPI integration, Pydantic v2 `model_dump_json` vs orjson, Redis/queue/log) | `references/python/orjson-stack.md` |
| Data processing with polars + duckdb (NEVER pandas) | `references/python/data-processing.md` |
| FastAPI + SQLAlchemy 2.x async stack | `references/python/fastapi-stack.md` |
| pydantic-ai agents | `references/python/pydantic-ai.md` |
| Textual TUI | `references/python/textual-tui.md` |
| Disposable PEP 723 scripts | `references/python/one-liners.md` |
| Canonical library defaults | `references/python/libraries.md` |

### Rust (`.rs`, `Cargo.toml`)

**READ `references/rust/README.md` FIRST.** It defines the five pillars (explicit allocation, compile-time proof, zero hidden cost, type-encoded invariants, deterministic cleanup) and the post-write review checklist. Then load on demand:

| Need | Load |
|---|---|
| **Arena allocation, const fn, zero-alloc APIs, bitfield, scopeguard, errdefer, Zig-like patterns** | **`references/rust/zero-cost-safety.md`** |
| Strict `Cargo.toml` lints + profile + workspace config | `references/rust/cargo-strict.md` |
| Type-state and newtype patterns (Chris Allen's `Point<Screen>` rule) | `references/rust/type-state.md` |
| `unsafe` discipline (safe wrapper + SAFETY comment + miri proof) | `references/rust/unsafe-discipline.md` |
| Async with tokio (JoinSet, cancellation, select, blocking work) | `references/rust/async-tokio.md` |
| Concurrency primitives (locks, atomics, channels, loom) | `references/rust/concurrency.md` |
| axum + sqlx + tracing + tower HTTP stack | `references/rust/axum-stack.md` |
| clap + color-eyre + tracing + indicatif CLI stack | `references/rust/clap-stack.md` |
| Property tests (proptest) + snapshot tests (insta) | `references/rust/proptest-insta.md` |
| Disposable `rust-script` scripts | `references/rust/one-liners.md` |
| Canonical library defaults | `references/rust/libraries.md` |
| **ANY `unsafe` / FFI / `MaybeUninit` / lock-free work** | **`references/rust-ub/` (full directory)** |

### TypeScript (`.ts`, `.tsx`, `.mts`, `.cts`)

**READ `references/typescript/README.md` FIRST.** Then load on demand:

| Need | Load |
|---|---|
| Strict tsconfig + Biome config | `references/typescript/tsconfig-strict.md` |
| Type patterns (branded types, `as const`, `satisfies`, narrowing, `assertNever`) | `references/typescript/type-patterns.md` |
| Data modeling (type vs interface vs Zod, readonly, parse-don't-validate) | `references/typescript/data-modeling.md` |
| Error handling (Result, typed errors, union vs throw, AbortSignal timeouts) | `references/typescript/error-handling.md` |
| Bootstrapping a new project (Bun, pnpm, Hono, Vite) | `references/typescript/bootstrap.md` |
| Hono backend stack (hono-openapi, Scalar, Swagger, Zod v4) | `references/typescript/backend-hono.md` |

### Go (`.go`, `go.mod`, `go.sum`, `.golangci.yml`, `*.proto`)

**READ `references/go/README.md` FIRST.** Then load on demand:

| Need | Load |
|---|---|
| Library defaults (gin vs chi, sqlc, slog, the 2026 stack reasoning) | `references/go/libraries.md` |
| Canonical strict `.golangci.yml` (v2) with per-linter rationale | `references/go/golangci-strict.md` |
| Project layout, Taskfile, CI, `go.mod` template | `references/go/bootstrap.md` |
| Type patterns (named types, smart constructors, sealed interfaces, generics) | `references/go/type-patterns.md` |
| Data modeling — the three layers of validation (validator/v10 → smart ctor → sqlc) | `references/go/data-modeling.md` |
| Error handling (`errors.Is/As`, typed errors, `%w` wrapping, no panic) | `references/go/error-handling.md` |
| Concurrency (`context.Context`, `errgroup`, channels, locks, `-race`, `goleak`) | `references/go/concurrency.md` |
| HTTP backend stack (gin + slog + validator + pgx, middleware ordering, SSE, WS) | `references/go/backend-stack.md` |
| RPC stack (Connect-Go default, grpc-go fallback, protovalidate, Buf) | `references/go/grpc-connect.md` |
| CLI stack (cobra + slog + huh) | `references/go/cobra-stack.md` |
| Database stack (sqlc + pgx + goose + testcontainers) | `references/go/sqlc-pgx.md` |
| TUI stack (bubbletea v2 + bubbles v2 + lipgloss v2; **CJK / IME support**) | `references/go/bubbletea-v2.md` |
| Testing (Given/When/Then, table-driven, fakes-over-mocks, autogold, rapid) | `references/go/testing.md` |
| Disposable `go run` scripts | `references/go/one-liners.md` |

---

## Activation

This skill activates whenever you are writing or modifying any `.py`, `.pyi`, `.rs`, `.ts`, `.tsx`, `.mts`, `.cts`, `.go` file, or any project manifest (`pyproject.toml`, `Cargo.toml`, `package.json`, `tsconfig.json`, `biome.json`, `go.mod`, `go.sum`, `.golangci.yml`, `Taskfile.yml`, `buf.yaml`, `sqlc.yaml`). Apply relevant language safety rules to one-off scripts too, with tooling and checks scaled by the scope contract above.

The references contain the recipes. **Read them before writing code. Re-read them when the model drifts.** The post-write review loop is non-negotiable.
