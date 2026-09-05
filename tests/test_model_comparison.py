from __future__ import annotations

from pathlib import Path
import inspect

import torch

from baselines.graph_tcn import GraphTCN, select_hidden_width
from baselines.lnln_subunit import LNPNLSubunit, select_subunit_count
from baselines.point_process_glm import PointProcessGLM
from evaluation.mechanistic_retina.rf_base import load_candidate0
from evaluation.mechanistic_retina.spike_banks import (
    generate_nested_spike_bank,
    slice_spike_bank,
)
from evaluation.model_comparison.artifacts import (
    load_comparison_checkpoint,
    save_comparison_checkpoint,
    validate_artifact_set,
)
from evaluation.model_comparison.config import load_comparison_config
from evaluation.model_comparison.parameters import parameter_inventory
from evaluation.model_comparison.prediction import masked_bernoulli_loss
from evaluation.model_comparison.rf import (
    conditional_total_dynamic_rf,
    evaluate_comparison_rf,
    glm_filter_rf,
)
from evaluation.model_comparison.runner_support import runner_contract
from evaluation.model_comparison.training import (
    BaselineTrainingRequest,
    train_glm_lbfgs,
)
from evaluation.v3_watchdog import WatchdogRequest
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.optimizer import phase1_parameters
from training.mechanistic_retina.stages import build_seed_data
from evaluation.mechanistic_retina.mechanism_runtime import build_student


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/model_comparison_t2.yaml"


def _canonical():
    config = load_comparison_config(CONFIG)
    candidate = load_candidate0(
        ROOT / config.candidate0_path,
        usage=config.candidate_teacher_usage,
        reference_candidate_index=config.candidate_teacher_reference_index,
    )
    data = build_seed_data(config.data_seed, candidate)
    return config, candidate, data


def _positions() -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...], tuple[str, ...]]:
    cones = torch.stack((torch.linspace(-0.2, 0.2, 29), torch.zeros(29)), dim=1)
    cells = torch.stack((torch.linspace(-0.15, 0.15, 16), torch.zeros(16)), dim=1)
    types = tuple("midget" if index < 8 else "parasol" for index in range(16))
    polarities = tuple("ON" if index % 8 < 4 else "OFF" for index in range(16))
    return cones, cells, types, polarities


def test_canonical_bank_identity_and_raw_current_architecture() -> None:
    config, candidate, data = _canonical()
    banks = {
        seed: slice_spike_bank(
            generate_nested_spike_bank(
                data.train_probability[:, 0],
                data.validation_probability[:, 0],
                seed=seed,
                max_trials=64,
            ),
            2,
        )
        for seed in config.bank_seeds
    }
    model = build_student(data, 19)
    assert all(
        (bank.train_sha256, bank.validation_sha256) == config.bank_hashes[seed]
        for seed, bank in banks.items()
    )
    assert model.config.architecture_mode.value == "mechanism_identifiable"
    assert float(model.gates.h1) == pytest.approx(0.01)
    assert candidate.rf_sha256 == config.candidate0_rf_sha256


def test_parameter_inventory_and_deterministic_matching() -> None:
    _, _, data = _canonical()
    model = build_student(data, 19)
    inventory = parameter_inventory(model, phase1_parameters(model))
    assert inventory.total == 264
    assert inventory.requires_grad == 264
    assert inventory.optimizer_listed == 136
    assert inventory.nonzero_gradient is None
    assert inventory.actually_updated is None
    assert select_subunit_count(264, 16, 4) == 2
    assert select_hidden_width(264, 16) == 5


def test_glm_lnln_and_graph_tcn_are_causal() -> None:
    cones_pos, cells_pos, types, polarities = _positions()
    models = (
        PointProcessGLM(29, 16, 16),
        LNPNLSubunit(cones_pos, cells_pos, types, polarities, 2),
        GraphTCN(cones_pos, cells_pos, 4),
    )
    first = torch.randn(2, 24, 29)
    second = first.clone()
    second[:, 13:] = torch.randn_like(second[:, 13:])
    history = torch.zeros(2, 24, 16)
    for model in models:
        left = model(first, history)
        right = model(second, history)
        assert torch.equal(left[:, :13], right[:, :13])


def test_glm_solver_reports_full_objective_convergence() -> None:
    torch.manual_seed(12)
    cones = torch.randn(8, 10, 2)
    targets = torch.sigmoid(0.8 * cones[..., :1] - 0.3)
    spikes = targets[:, None]
    mask = torch.ones_like(spikes, dtype=torch.bool)
    model = PointProcessGLM(2, 1, 2)
    result = train_glm_lbfgs(
        BaselineTrainingRequest(
            model,
            model,
            cones,
            spikes,
            mask,
            100,
            (0, 100),
            0.03,
            8,
            12,
        )
    )
    assert result.converged
    assert result.checkpoints[-1].train_nll < result.checkpoints[0].train_nll


def test_lnln_locality_shape_and_graph_receptive_field() -> None:
    cones, cells, types, polarities = _positions()
    lnln = LNPNLSubunit(cones, cells, types, polarities, 2)
    graph = GraphTCN(cones, cells, 4)
    stimulus = torch.randn(3, 20, 29)
    history = torch.zeros(3, 20, 16)
    assert lnln(stimulus, history).shape == (3, 20, 16)
    assert bool((lnln.spatial_kernels[~lnln.local_support] == 0).all())
    assert graph(stimulus, history).shape == (3, 20, 16)
    assert graph.receptive_field_steps >= 16
    assert bool((graph.cell_pool[~graph.cell_support] == 0).all())


def test_all_models_use_the_same_bernoulli_loss_without_rf_target() -> None:
    logits = torch.randn(2, 5, 3)
    targets = torch.rand(2, 5, 3)
    mask = torch.ones_like(targets, dtype=torch.bool)
    assert torch.equal(
        masked_bernoulli_loss(logits, targets, mask),
        expected_bernoulli_nll(logits, targets, mask),
    )
    assert "rf_target" not in inspect.signature(BaselineTrainingRequest).parameters


def test_unified_rf_matches_glm_parameter_filter() -> None:
    torch.manual_seed(4)
    model = PointProcessGLM(3, 2, 4)
    with torch.no_grad():
        model.kernel.normal_()
    cones = torch.randn(2, 7, 3)
    history = torch.zeros(2, 7, 2)
    jacobian = conditional_total_dynamic_rf(model, cones, history, 4)
    parameter_rf = glm_filter_rf(model, jacobian.shape[0])
    assert torch.allclose(jacobian, parameter_rf)


def test_exact_nearest_type_and_prototype_metrics_are_distinct() -> None:
    _, candidate, data = _canonical()
    result = evaluate_comparison_rf(
        candidate.rf.unsqueeze(0).expand(12, -1, -1, -1).clone(),
        candidate,
        data.cone_positions,
        data.cell_positions,
    )
    assert result.exact_fraction == 1.0
    assert result.nearest_type_polarity_fraction == 1.0
    assert (
        result.prototype_centroid_fraction
        == result.summary.metric.type_polarity_fraction
    )


def test_checkpoint_roundtrip_and_watchdog_contract(tmp_path: Path) -> None:
    cones, cells, _, _ = _positions()
    model = GraphTCN(cones, cells, 4)
    path = tmp_path / "final.pt"
    saved = save_comparison_checkpoint(path, model, {"run_id": "test", "step": 400})
    restored = GraphTCN(cones, cells, 4)
    metadata = load_comparison_checkpoint(path, restored)
    assert saved.bytes > 0 and metadata["run_id"] == "test"
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            model.state_dict().values(), restored.state_dict().values(), strict=True
        )
    )
    contract = runner_contract(ROOT)
    assert isinstance(contract.watchdog, WatchdogRequest)
    assert contract.watchdog.monitor_interval_seconds == 300.0
    assert contract.watchdog.stall_intervals == 2


def test_strict_artifact_contract(tmp_path: Path) -> None:
    required = runner_contract(ROOT).required_evidence
    for name in required:
        path = tmp_path / name
        payload = (
            b"PNG"
            if name == "pareto.png"
            else b"{}" if name.endswith((".json", ".jsonl", ".yaml")) else b"x\n"
        )
        path.write_bytes(payload)
    validate_artifact_set(tmp_path, required)
