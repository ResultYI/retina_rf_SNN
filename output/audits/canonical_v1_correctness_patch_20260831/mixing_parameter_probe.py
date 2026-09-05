# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.6.0"]
# ///
# Run from repository root with D:/anaconda/python.exe -B <this-file>.
# Counts only; constructs no optimizer and performs no parameter updates.
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path.cwd()))
from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.model import build_mechanistic_retina


root = Path(__file__).resolve().parent
old_path = root / "source_before/models/mechanistic_retina/shared_subunits.py"
spec = importlib.util.spec_from_file_location("audit_original_shared_subunits", old_path)
assert spec is not None and spec.loader is not None
old_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old_module
spec.loader.exec_module(old_module)
axis = torch.arange(-6, 7, dtype=torch.float64) * 0.04
cones = torch.cartesian_prod(axis, axis)
fixtures = (
    (
        "original_audit_N6_E10",
        ((0, 0), (0.04, 0), (0, 0.04), (0, 0), (0.04, 0), (0, 0.04)),
        ("midget", "midget", "midget", "parasol", "parasol", "parasol"),
        ("ON", "ON", "OFF", "ON", "ON", "OFF"),
        None,
    ),
    (
        "all_self_only_N4_E4",
        ((0, 0),) * 4,
        ("midget", "midget", "parasol", "parasol"),
        ("ON", "OFF", "ON", "OFF"),
        None,
    ),
    (
        "mixed_degree_N3_E5",
        ((0, 0), (0.04, 0), (0.12, 0)),
        ("midget",) * 3,
        ("ON",) * 3,
        torch.tensor(((0, 0, 1, 1, 2), (0, 1, 0, 1, 2))),
    ),
    ("fixed_N1_E1", ((0, 0),), ("midget",), ("ON",), None),
)
rows = []
for name, positions, cell_types, polarities, edges in fixtures:
    cells = torch.tensor(positions, dtype=torch.float64)
    config = MechanisticRetinaConfig(architecture_mode="mechanism_identifiable")
    model = build_mechanistic_retina(
        config, cones, cells, cell_types, polarities, shared_subunit_edge_index=edges
    )
    before_mixer = old_module.SharedSubunitMixer(
        old_module.SharedSubunitLayout(cells, cell_types, polarities, edges),
        radius_deg=config.shared_subunit_radius_deg,
        trainable=True,
    ).double()
    after_mixer = model.shared_subunits
    before_mix_parameters = tuple(value for value in before_mixer.parameters() if value.requires_grad)
    after_mix_parameters = tuple(value for value in after_mixer.parameters() if value.requires_grad)
    before_mix_scalars = sum(value.numel() for value in before_mix_parameters)
    after_mix_scalars = sum(value.numel() for value in after_mix_parameters)
    after_parameters = tuple(value for value in model.parameters() if value.requires_grad)
    after_total = sum(value.numel() for value in after_parameters)
    degree = torch.bincount(after_mixer.edge_index[0], minlength=len(positions))
    before_count = after_total - after_mix_scalars + before_mix_scalars
    before_tensors = len(after_parameters) - len(after_mix_parameters) + len(before_mix_parameters)
    row = {
        "fixture": name,
        "cell_count": len(positions),
        "edge_count": after_mixer.edge_index.shape[1],
        "self_only_rows": int((degree == 1).sum()),
        "mixing_scalars_before": before_mix_scalars,
        "mixing_scalars_after": after_mix_scalars,
        "trainable_scalars_before": before_count,
        "trainable_scalars_after": after_total,
        "trainable_tensors_before": before_tensors,
        "trainable_tensors_after": len(after_parameters),
        "initial_connection_matrix_bitwise_equal": torch.equal(
            before_mixer.connection_matrix(), after_mixer.connection_matrix()
        ),
        "mixing_raw_state_shape_before": list(before_mixer.raw_connections.shape),
        "mixing_raw_state_shape_after": list(after_mixer.raw_connections.shape),
    }
    assert row["initial_connection_matrix_bitwise_equal"]
    assert before_mix_scalars - after_mix_scalars == (int((degree == 1).sum()) if len(positions) > 1 else 0)
    rows.append(row)
payload = {
    "original_mixer_source": str(old_path),
    "before_total_method": "current unchanged non-mixer parameter count plus original snapshot mixer count",
    "training": False,
    "optimizer_constructed": False,
    "fixtures": rows,
}
serialized = json.dumps(payload, indent=2)
(root / "mixing_parameter_counts.json").write_text(serialized + "\n", encoding="utf-8")
print(serialized)
