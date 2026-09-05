from __future__ import annotations

from dataclasses import asdict

from data.karamanlis_rf_population import RFPopulationMarmosetData
from models.mechanistic_retina.contracts import (
    MECHANISTIC_MODEL_REVISION,
    MechanisticRetinaConfig,
)


def rf_population_checkpoint_base(
    data: RFPopulationMarmosetData,
    model_config: MechanisticRetinaConfig,
    *,
    training_seed: int,
):
    return {
        "schema": "karamanlis_2024_marmoset_rf_geometry_canonical_retina_v1",
        "revision": MECHANISTIC_MODEL_REVISION,
        "session_id": data.session_id,
        "model_config": asdict(model_config)
        | {"architecture_mode": model_config.architecture_mode.value},
        "cell_ids": data.cell_ids,
        "cell_types": data.cell_types,
        "polarities": data.polarities,
        "model_cell_positions": data.model_cell_positions,
        "model_cone_positions": data.model_cone_positions,
        "cell_positions_um": data.cell_positions_um,
        "cone_positions_um": data.cone_positions_um,
        "cone_blocks_screen_indices": data.cone_blocks_screen_indices,
        "edge_index": data.edge_index,
        "training_seed": training_seed,
    }


__all__ = ["rf_population_checkpoint_base"]
