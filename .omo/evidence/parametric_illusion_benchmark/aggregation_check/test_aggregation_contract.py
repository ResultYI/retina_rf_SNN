#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///

# ─── How to run ───
# 1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
# 2. Run: uv run test_aggregation_contract.py
# ──────────────────

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from aggregation_core import AggregationSpec, Stratum, control_subtracted, point_estimate, polarity  # noqa: E402


def main() -> None:
    # Given: four class means with deliberately unequal cell counts.
    values = np.asarray([1.0] * 5 + [-3.0] * 4 + [5.0] * 9 + [-7.0] * 4)
    on = np.asarray(tuple(range(5)) + tuple(range(9, 18)))
    off = np.asarray(tuple(range(5, 9)) + tuple(range(18, 22)))
    groups = tuple(np.asarray(indices) for indices in (
        range(0, 5), range(5, 9), range(9, 18), range(18, 22)
    ))

    # When: the three declared cohort weighting contracts are evaluated.
    raw = point_estimate(values, AggregationSpec("raw22", (Stratum(np.arange(22), 1.0),)))
    polarity_equal = point_estimate(values, AggregationSpec(
        "polarity_equal", (Stratum(on, 0.5), Stratum(off, 0.5))
    ))
    group_equal = point_estimate(values, AggregationSpec(
        "four_group_equal", tuple(Stratum(indices, 0.25) for indices in groups)
    ))
    subtracted = control_subtracted(np.asarray((10.0, 13.0, 8.0)))

    # Then: weighting and width-zero subtraction have their exact meanings.
    assert polarity("MC_ON") == "ON" and polarity("PC_OFF") == "OFF"
    assert np.isclose(raw, 10.0 / 22.0)
    assert np.isclose(polarity_equal, -5.0 / 7.0)
    assert np.isclose(group_equal, -1.0)
    assert np.array_equal(subtracted, np.asarray((0.0, 3.0, -2.0)))
    print("PASS aggregation weighting and width-zero subtraction contract")


if __name__ == "__main__":
    main()
