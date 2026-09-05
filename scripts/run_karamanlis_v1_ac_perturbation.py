#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py", "numpy", "torch"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run scripts/run_karamanlis_v1_ac_perturbation.py
# 3. Or make executable and run:
#      chmod +x scripts/run_karamanlis_v1_ac_perturbation.py && ./scripts/run_karamanlis_v1_ac_perturbation.py
# ──────────────────

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.mechanistic_retina.karamanlis_v1_ac_perturbation import (
    V1ACPerturbationConfig,
    run_v1_ac_perturbation,
)


def main() -> None:
    result = run_v1_ac_perturbation(
        V1ACPerturbationConfig(
            session_dir=Path(
                "data/real/karamanlis_2024/sessions/20220301_252MEA_marmoset_left_n1"
            ),
            graph_dir=Path(
                "output/real_data/karamanlis_2024_population_locality_graph_v1"
            ),
            checkpoint_path=Path(
                "output/real_data/karamanlis_2024_population_rf_geometry_cell_gains_seed20260302/model-best.pt"
            ),
            output_dir=Path("output/real_data/karamanlis_2024_v1_ac_perturbation_v3"),
        )
    )
    print(result)


if __name__ == "__main__":
    main()
