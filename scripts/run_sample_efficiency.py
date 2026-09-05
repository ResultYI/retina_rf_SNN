#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pillow", "torch"]
# ///

# ─── How to run ───
# 1. Validate the frozen no-training contract:
#      uv run scripts/run_sample_efficiency.py --validate-only
# 2. Execute the bounded experiment:
#      uv run scripts/run_sample_efficiency.py
# 3. Resume complete matching fraction caches:
#      uv run scripts/run_sample_efficiency.py --resume
# ──────────────────

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.model_comparison.sample_efficiency_runner import (
    RunnerRequest,
    SampleEfficiencyRunnerError,
    build_validation_contract,
    fixture_row_provider,
    run_sample_efficiency,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_comparison_sample_efficiency_t2.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--fixture-mode", action="store_true")
    parser.add_argument("--hold-lock-seconds", type=float, default=0.0)
    args = parser.parse_args()
    config_path = Path(args.config)
    try:
        if args.validate_only:
            print(json.dumps(build_validation_contract(_REPO_ROOT, config_path), allow_nan=False, sort_keys=True))
            return 0
        result = run_sample_efficiency(
            RunnerRequest(
                _REPO_ROOT,
                config_path,
                resume=args.resume,
                row_provider=fixture_row_provider if args.fixture_mode else None,
                hold_lock_seconds=args.hold_lock_seconds,
            )
        )
        print(json.dumps({"output_dir": str(result.output_dir), "artifact_sha256": dict(result.artifact_sha256)}, allow_nan=False, sort_keys=True))
        return 0
    except SampleEfficiencyRunnerError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
