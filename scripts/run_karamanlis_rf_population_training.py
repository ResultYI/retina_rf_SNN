# /// script
# requires-python = ">=3.11"
# dependencies = ["h5py", "numpy", "torch"]
# ///

from pathlib import Path

from evaluation.mechanistic_retina.karamanlis_rf_population_run import (
    RFPopulationRunConfig,
    run_rf_population_training,
)


SESSION = Path(
    "data/real/karamanlis_2024/sessions/20220301_252MEA_marmoset_left_n1"
)
GRAPH = Path(
    "output/real_data/karamanlis_2024_population_locality_graph_v1"
)
OUTPUT = Path(
    "output/real_data/karamanlis_2024_population_rf_geometry_cell_gains_seed20260302"
)


def main() -> None:
    result = run_rf_population_training(
        RFPopulationRunConfig(
            session_dir=SESSION,
            graph_dir=GRAPH,
            output_dir=OUTPUT,
        )
    )
    print(result)


if __name__ == "__main__":
    main()
