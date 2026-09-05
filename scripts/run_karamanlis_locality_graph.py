#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py", "numpy", "torch"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run from the repository root (no venv or pip install needed):
#      uv run python -m scripts.run_karamanlis_locality_graph
# 3. Or use the Retina project environment:
#      python -m scripts.run_karamanlis_locality_graph
# ──────────────────

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from evaluation.mechanistic_retina.karamanlis_locality_artifacts import (
    run_karamanlis_locality_graph,
)


RF_SOURCE: Final = Path(
    "output/real_data/karamanlis_2024_population_rf_centers_v1"
)
OUTPUT: Final = Path(
    "output/real_data/karamanlis_2024_population_locality_graph_v1"
)


def main() -> None:
    result = run_karamanlis_locality_graph(RF_SOURCE, OUTPUT)
    payload = json.loads(result.results_path.read_text(encoding="utf-8"))
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
