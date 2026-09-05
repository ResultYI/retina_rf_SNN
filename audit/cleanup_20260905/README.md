# Repository cleanup for independent review — 2026-09-05

## Scope and recovery

No model, loader, training, loss, bounds, split or baseline implementation was
changed. No experiment was run. All existing Python sources/tests were kept;
README/CURRENT_STATE and the audit navigation now identify the current shared-BC
lineage instead of the old 50-step lineage.

Before cleanup: Git HEAD `7dc00b1e9f2902186891a97c022e5d04fab6d1d4`, branch
`rgc-readout-v2`, remote `https://github.com/ResultYI/retina_rf_SNN.git`.
The working tree already contained many uncommitted/untracked research changes.
They were not reset or committed. Git history alone cannot recover these files.

Archive: `.local_archives/20260905-pre-audit-cleanup/recoverable-content.zip`.
Its 1,310 entries include every removed file and the scientific-source snapshot;
every entry was read back and verified by length and SHA256 before removal.
The archive also has the pre-cleanup Git status and tracked diff alongside it.
See `backup_verification.json`, `source_before.csv` and `archived_files.csv`.

879 files (329,126,066 bytes; 313.88 MiB) were removed from active paths.
The recovery ZIP occupies 113,998,162 bytes (108.72 MiB); net reduction before
new small audit documentation is 215,127,904 bytes (205.16 MiB).
Extract an individual archived relative path into a temporary directory and
compare it before restoring; do not overwrite newer working files blindly.

## What was removed and what stayed

Removed categories: unreferenced older synthetic recovery/spike-budget runs,
duplicate/obsolete marmoset AC perturbation and early fit artifacts, old
Candidate0/readout/manual-QA outputs, temporary fixture results, historical
presentation reports and an older code-review package. The full exact list is
`archived_files.csv`. No result was selected for removal based on its score/sign.

Preserved: current shared-BC macaque fits, current synthetic sanity, final
prediction baselines and SC correction history, RF/pathway outputs, frozen
applications, parametric/aggregation evidence, independent seeds, timing
uncertainty, multi-spike/reset checks, correctness/history audits, and separate
marmoset RF geometry and V1 reference results.

Literal dependencies from Python/configs/tests and retained artifact producers
were followed conservatively. `retained_dependencies.csv` records why older
bundles remain. Some old drivers, including V2 and fixed-step drivers, still
refer to their historical artifacts: both were kept rather than deleting
source or changing a numerical contract. Older material is not the primary
audit target; use the root index. Several inaccessible temporary directories
were left untouched, listed in `inaccessible_retained.txt`.

## Git publication

`.gitignore` now exposes curated `.omo/evidence` results and final small
checkpoints. Raw recordings, large duplicate input banks, cache directories and
the recovery archive stay local. `publish_manifest.csv` lists intended
Git-visible files/hashes; `local_only_artifacts.csv` lists retained excluded
evidence with hashes. The publication checker rejects any single file >=100 MiB
and checks essential evidence visibility. It does not stage, commit or push.

`.gitattributes` preserves source/config and immutable evidence bytes across
platform checkouts so newline conversion does not invalidate provenance hashes.
Do not replace actual scientific-source hashes with Git HEAD when reviewing runs.

## Verification

- 402 scientific source/config/test files remained SHA256-identical.
- Focused existing no-training contracts: **127 passed, 4 deselected**.
  Includes shared-BC/clamp/RF, support/metadata, parameter bounds, macaque and
  marmoset adapters, LN/CNN forward and causality checks.
- `focused_tests.txt` and `focused_tests.xml` contain the actual test outcome.
- `publication_verification.json` records final evidence visibility, 29 consumed
  prediction-source checks and size limits. Read its actual status, not this
  prose alone.
- All 22 Canonical, 22 LN and 22 CNN final checkpoints are Git-visible.
  Publication is approximately 240 MiB; largest individual file is 27.74 MiB.
  Duplicate pre/post-correctness regression tensor snapshots and historical
  comparator figures remain local with hashes rather than expanding the review.
- All Markdown links in the five new/updated entry documents resolve locally.

The prompt to paste into Pro is `audit/PRO_AUDIT_PROMPT_ZH.md`. Review and commit
the current branch yourself, then point Pro at the exact uploaded branch/commit.
This task does not assert that a model/results audit has already passed.
