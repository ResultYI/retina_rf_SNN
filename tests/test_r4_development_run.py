from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch

from evaluation.mechanistic_retina.schottdorf_ln_source import LNSourcePaths


def test_given_benchmark_when_loading_then_cell_entrypoint_exists() -> None:
    assert importlib.util.find_spec("evaluation.mechanistic_retina.schottdorf_r4_development") is not None


def test_given_frozen_output_when_running_then_rejected_before_loading() -> None:
    module = importlib.import_module("evaluation.mechanistic_retina.schottdorf_r4_development")
    source = Path("output/real_data/schottdorf_lee_2021_22cell_canonical_v1_revision4_fresh_20260829")
    paths = LNSourcePaths(Path("unused"), Path("unused"), source)
    with pytest.raises(ValueError, match="outside frozen"):
        module.run_r4_development_cell(paths, "67#4", source)


def test_fresh_state_allows_only_fixed_graph_roundoff() -> None:
    module = importlib.import_module("evaluation.mechanistic_retina.schottdorf_r4_development")
    reference = {"h1.graph.edge_weight": torch.tensor([0.04]), "weight": torch.tensor([1.0])}
    current = {name: value.clone() for name, value in reference.items()}
    current["h1.graph.edge_weight"] += 2e-7
    assert module.verify_fresh_state(current, reference) > 0
    current["weight"] += 2e-7
    with pytest.raises(ValueError, match="weight"):
        module.verify_fresh_state(current, reference)
    current["weight"] = reference["weight"]
    current["h1.graph.edge_weight"] += 1e-3
    with pytest.raises(ValueError, match="h1.graph.edge_weight"):
        module.verify_fresh_state(current, reference)
