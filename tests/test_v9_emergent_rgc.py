from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import torch
import yaml

import evaluation.v9 as evaluation_v9
import training.v9 as training_v9
from data.dataset import fit_log_cone_stats
from models.v9_retina import (
    AnonymousRGCOutput,
    build_v9_model,
    detach_v9_state,
)
from scripts.run_v9_emergent_rgc_dynamic_rf import (
    _load_evaluation_weights,
    _write_failure,
)
from training.v9 import (
    EnergyBudgetState,
    V9TrainConfig,
    V9Trainer,
    prepare_data,
)


def _positions() -> np.ndarray:
    return np.asarray(
        ((0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.0, 0.1), (0.1, 0.1), (0.2, 0.1)),
        dtype=np.float32,
    )


def _config() -> V9TrainConfig:
    return V9TrainConfig(
        sequence_steps=8,
        burn_in_steps=2,
        differentiable_steps=6,
        context_only_steps=4,
        supervised_steps=2,
        checkpoint_block_steps=2,
        gradient_accumulation_steps=1,
        max_optimizer_steps=2,
        min_optimizer_steps=1,
        validation_interval_steps=1,
        early_stopping_patience_validations=1,
        energy_bootstrap_fraction=0.5,
    )


def _trainer(seed: int = 19) -> V9Trainer:
    core, decoder = build_v9_model(
        _positions(), dt_ms=5.0, eccentricity_deg=4.0, seed=seed
    )
    return V9Trainer(core, decoder, _config(), reconstruction_scale=1.0)


def test_anonymous_banks_are_structurally_symmetric() -> None:
    trainer = _trainer()
    encoder = trainer.core.rgc
    assert encoder.unit_centers_degs.shape == (6, 2)
    assert encoder.spatial_pools[0].indices().equal(encoder.spatial_pools[1].indices())
    for name in encoder._PARAMETERS:
        values = encoder.bounded(name)
        assert values.shape == (2,)
        assert torch.allclose(values[0], values[1], rtol=0.05, atol=0.02)


def test_v9_parameter_names_have_no_population_identity_prior() -> None:
    trainer = _trainer()
    names = tuple(name.lower() for name, _ in trainer.core.rgc.named_parameters())
    assert not any("midget" in name or "parasol" in name for name in names)
    mix = trainer.core.rgc.bounded("mix")
    assert torch.all((mix > 0) & (mix < 1))


def test_tied_decoder_has_no_spatial_parameters() -> None:
    trainer = _trainer()
    assert set(dict(trainer.decoder.named_parameters())) == {"raw_magnitude", "bias"}


def test_reconstruction_gradient_reaches_encoder_spatial_parameters() -> None:
    trainer = _trainer()
    sequence = torch.full((1, 8, 6), 2.0)
    output, _ = trainer.core.forward_sequence(sequence)
    trainer.decoder(output).square().mean().backward()
    assert trainer.core.rgc.raw_spatial_residual.grad is not None
    assert trainer.core.rgc.raw_sigma.grad is not None
    assert torch.isfinite(trainer.core.rgc.raw_spatial_residual.grad).all()
    assert torch.isfinite(trainer.core.rgc.raw_sigma.grad).all()


def test_burn_in_path_detaches_state_exactly_once(monkeypatch) -> None:
    trainer = _trainer()
    count = 0
    original = training_v9.detach_v9_state

    def counted(state):
        nonlocal count
        count += 1
        return original(state)

    monkeypatch.setattr(training_v9, "detach_v9_state", counted)
    noisy = torch.randn(1, 8, 6)
    trainer.forward_clip(noisy, noisy, checkpointed=False)
    assert count == 1


def test_activation_checkpoint_matches_plain_forward() -> None:
    trainer = _trainer()
    noisy = torch.randn(1, 8, 6)
    initial = trainer.core.initial_state(1, torch.device("cpu"))
    with torch.no_grad():
        _, state = trainer.core.forward_sequence(noisy[:, :2], initial)
    state = detach_v9_state(state)
    plain, plain_state = trainer._forward_region(noisy[:, 2:], state, False)
    checked, checked_state = trainer._forward_region(noisy[:, 2:], state, True)
    torch.testing.assert_close(plain.spikes, checked.spikes)
    torch.testing.assert_close(plain.rates, checked.rates)
    torch.testing.assert_close(plain_state.rgc.rate, checked_state.rgc.rate)


def test_energy_penalty_is_zero_below_budget() -> None:
    state = EnergyBudgetState(1, 0.9, 0.01, 10.0, budget=1.0, ema=1.0)
    penalty, violation = state.penalty(torch.tensor(0.5), rho=1.0)
    assert penalty.item() == 0.0
    assert violation.item() == 0.0


def test_dual_state_round_trips_through_checkpoint_payload() -> None:
    trainer = _trainer()
    trainer.energy_state.update(0.5)
    generator = torch.Generator().manual_seed(1)
    payload = trainer.checkpoint_payload(generator)
    restored = _trainer()
    other_generator = torch.Generator()
    restored.restore(payload, other_generator)
    assert restored.energy_state == trainer.energy_state
    assert torch.equal(other_generator.get_state(), generator.get_state())


def test_evaluation_checkpoint_preserves_terminal_energy_state(tmp_path: Path) -> None:
    terminal = _trainer()
    terminal.energy_state.update(0.5)
    terminal.energy_state.update(0.4)
    terminal_state = terminal.energy_state
    terminal_step = terminal.optimizer_step
    selected = _trainer(seed=23)
    checkpoint = tmp_path / "selected.pt"
    torch.save(selected.checkpoint_payload(torch.Generator().manual_seed(23)), checkpoint)

    _load_evaluation_weights(terminal, checkpoint, torch.device("cpu"))

    assert terminal.energy_state is terminal_state
    assert terminal.optimizer_step == terminal_step
    assert torch.equal(
        terminal.core.rgc.raw_sigma,
        selected.core.rgc.raw_sigma,
    )


def test_reconstruction_loss_uses_only_supervised_timepoints() -> None:
    trainer = _trainer()
    noisy = torch.randn(1, 8, 6)
    clean = torch.randn(1, 8, 6)
    first, _, _ = trainer.forward_clip(noisy, clean, checkpointed=False)
    changed = clean.clone()
    changed[:, :-2] += 100.0
    second, _, _ = trainer.forward_clip(noisy, changed, checkpointed=False)
    torch.testing.assert_close(first.raw_reconstruction, second.raw_reconstruction)


def test_validation_reuses_train_normalization_statistics() -> None:
    train = tuple(sorted(Path("data/isetbio_bsds300_4deg/train").glob("*.h5")))
    validation = tuple(sorted(Path("data/isetbio_bsds300_4deg/val").glob("*.h5")))
    prepared = prepare_data(train, validation)
    mean, scale = fit_log_cone_stats(train)
    np.testing.assert_allclose(prepared.normalization_mean, mean)
    np.testing.assert_allclose(prepared.normalization_scale, scale)


def test_local_linear_baseline_is_disabled_by_default() -> None:
    config = yaml.safe_load(
        Path("configs/v9_emergent_rgc_dynamic_rf.yaml").read_text(encoding="utf-8")
    )
    assert config["local_linear_baseline"] == "not_run"


def test_v9_evaluation_has_no_dense_jacobian_or_local_normal_equation() -> None:
    source = inspect.getsource(evaluation_v9)
    assert "autograd.functional.jacobian" not in source
    assert "fit_local_linear_baseline" not in source


def test_smoke_step_has_finite_nonzero_temporal_gradients() -> None:
    trainer = _trainer()
    noisy = torch.randn(1, 8, 6) + 1.0
    clean = torch.randn(1, 8, 6)
    result = trainer.train_optimizer_step(((noisy, clean),))
    assert result.temporal_gradient_norm > 0
    assert np.isfinite(result.temporal_gradient_norm)


def test_both_anonymous_banks_have_nonzero_activity() -> None:
    trainer = _trainer()
    sequence = torch.full((1, 32, 6), 10.0)
    output, _ = trainer.core.forward_sequence(sequence)
    assert torch.all(output.spikes.sum(dim=(0, 1, 3)) > 0)


def test_failure_path_writes_blocker_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("seed: 19\n", encoding="utf-8")
    args = argparse.Namespace(config=config, device="cuda", seed=19)
    _write_failure(run_dir, args, {"seed": 19}, RuntimeError("boom"), "stamp")
    status = json_load(run_dir / "run_status.json")
    assert status["status"] == "FAILED_WITH_REPORT"
    assert (run_dir / "blocker_report_zh.md").is_file()


def json_load(path: Path) -> dict[str, str]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
