#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.2.6", "pydantic==2.8.2"]
# ///
# How to run: D:/anaconda/python.exe -B .omo/evidence/final_prediction_results/check_statistics.py
from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np


def main() -> None:
    namespace = runpy.run_path(str(Path(__file__).with_name("build_package.py")))
    summarize = namespace["paired_summary"]
    # Given a constant paired difference, bootstrap uncertainty must collapse exactly.
    difference = np.full(22, -0.25)
    indices = np.random.default_rng(7).integers(0, 22, (100, 22))
    result = summarize(difference, indices)
    assert result.mean == result.median == result.mean_ci_low == result.mean_ci_high == -0.25
    assert result.median_ci_low == result.median_ci_high == -0.25
    assert (result.first_wins, result.second_wins, result.ties) == (22, 0, 0)
    # Given complete pair preservation, identical models have only ties and zero differences.
    result = summarize(np.zeros(22), indices)
    assert result.mean == result.median == result.mean_ci_low == result.mean_ci_high == 0
    assert (result.first_wins, result.second_wins, result.ties) == (0, 0, 22)
    print("PASS: paired constant-difference and exact-tie checks; no model execution")


if __name__ == "__main__":
    main()
