#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py", "numpy", "torch"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run scripts/run_karamanlis_v1_rf_validation.py
# 3. Or make executable and run:
#      chmod +x scripts/run_karamanlis_v1_rf_validation.py && ./scripts/run_karamanlis_v1_rf_validation.py
# ─────────────────

from __future__ import annotations

from pathlib import Path

from evaluation.mechanistic_retina.karamanlis_v1_rf_validation import (
    V1RFValidationConfig,
    run_v1_rf_validation,
)


def main() -> None:
    result = run_v1_rf_validation(
        V1RFValidationConfig(
            session_dir=Path(
                "data/real/karamanlis_2024/sessions/20220301_252MEA_marmoset_left_n1"
            ),
            graph_dir=Path(
                "output/real_data/karamanlis_2024_population_locality_graph_v1"
            ),
            sta_dir=Path("output/real_data/karamanlis_2024_population_rf_centers_v1"),
            checkpoint_path=Path(
                "output/real_data/karamanlis_2024_population_rf_geometry_cell_gains_seed20260302/model-best.pt"
            ),
            output_dir=Path(
                "output/real_data/karamanlis_2024_v1_independent_rf_validation_v1"
            ),
        )
    )
    print(result["summary"])


if __name__ == "__main__":
    main()
