from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_experiment import ResponseExperimentError, _prepare_output


ROOT = Path(__file__).resolve().parents[1]


def test_final_test_requires_diagnostics_only_checkpoint(tmp_path: Path) -> None:
    args = argparse.Namespace(
        output=str(tmp_path / "final"),
        diagnostics_only=False,
        final_test=True,
        checkpoint="checkpoint.pt",
        resume=None,
        overwrite=False,
    )

    with pytest.raises(ResponseExperimentError, match="diagnostics-only"):
        _prepare_output(args)


def test_run_experiment_cli_exposes_final_test() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_experiment.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--final-test" in result.stdout
