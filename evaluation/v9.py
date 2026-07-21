from __future__ import annotations

# noqa: SIZE_OK — one bounded evaluation/report pipeline with no reusable subservice.

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from models.v9_retina import AnonymousRGCOutput, V9RetinaCore, detach_v9_state
from training.v9 import PreparedData, V9Trainer


def evaluate_v9(
    trainer: V9Trainer,
    data: PreparedData,
    validation_clips: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    context_pairs: int,
    lag_steps: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    validation = trainer.evaluate(validation_clips)
    targets = torch.cat([clean[:, -trainer.config.supervised_steps :] for _, clean in validation_clips])
    zero_mse = float(targets.square().mean())
    train_mean = float(torch.cat(data.train).mean())
    global_mse = float((targets - train_mean).square().mean())
    best_baseline = min(zero_mse, global_mse)
    reconstruction = {
        "validation_raw_current_mse": validation["raw_reconstruction"],
        "validation_normalized_current_mse": validation["normalized_reconstruction"],
        "zero_baseline_mse": zero_mse,
        "train_fit_global_mean_baseline_mse": global_mse,
        "skill_score": 1.0 - validation["raw_reconstruction"] / best_baseline,
        "local_linear_baseline": {
            "status": "not_run",
            "reason": "disabled_by_default_to_avoid_high_dimensional_normal_equations",
            "estimated_feature_count_per_target": 1
            + trainer.config.sequence_steps
            * int(torch.bincount(trainer.core.rgc.support_indices[0]).float().mean()),
        },
    }
    bank_features = _bank_features(trainer, validation_clips, data.dt_ms)
    dynamic_rf = _dynamic_rf(
        trainer.core,
        data.validation,
        context_pairs=context_pairs,
        lag_steps=lag_steps,
        dt_ms=data.dt_ms,
        seed=seed,
        device=device,
    )
    bounds = _parameter_bounds(trainer)
    return {
        "reconstruction": reconstruction,
        "resources": {
            "energy": validation["energy"],
            "budget": validation["energy_budget"],
            "violation": validation["energy_violation"],
            "dual": trainer.energy_state.dual,
            "bank_polarity_rates": validation["bank_polarity_rates"],
            "active_unit_fraction": validation["active_unit_fraction"],
        },
        "bank_features": bank_features,
        "dynamic_rf": dynamic_rf,
        "parameter_bounds": bounds,
    }


def _bank_features(
    trainer: V9Trainer,
    validation_clips: Sequence[tuple[torch.Tensor, torch.Tensor]],
    dt_ms: float,
) -> dict[str, Any]:
    encoder = trainer.core.rgc
    unit_rates = []
    with torch.no_grad():
        for noisy, _ in validation_clips:
            _, history, _ = trainer.forward_clip(noisy, noisy, checkpointed=False)
            unit_rates.append(history.rates.mean(dim=(0, 1, 3)).cpu())
    mean_unit_rate = torch.stack(unit_rates).mean(dim=0)
    unit_radius = []
    for pool in encoder.spatial_pools:
        row_cost = pool.values().new_zeros(pool.shape[0]).scatter_add(
            0, pool.indices()[0], pool.values() * encoder.distance_sq_degs
        )
        unit_radius.append(row_cost.sqrt().detach().cpu())
    unit_radius_tensor = torch.stack(unit_radius)
    temporal = _temporal_features(trainer.core, dt_ms)
    mix = encoder.bounded("mix").detach().cpu()
    membrane_tau = encoder.bounded("membrane_tau_ms").detach().cpu()
    adaptation_tau = encoder.bounded("adaptation_tau_ms").detach().cpu()
    adaptation_gain = encoder.bounded("adaptation_gain").detach().cpu()
    rate_tau = encoder.bounded("rate_tau_ms").detach().cpu()
    amacrine_gain = encoder.bounded("amacrine_gain").detach().cpu()
    threshold = encoder.bounded("threshold").detach().cpu()
    sigma = encoder.sigma_degs.detach().cpu()
    decoder_magnitude = trainer.decoder.magnitude.detach().cpu()
    features = torch.stack(
        (
            unit_radius_tensor,
            mix[:, None].expand_as(unit_radius_tensor),
            mean_unit_rate,
        ),
        dim=-1,
    )
    separation = _silhouette(features)
    radius_diff, radius_ci = _paired_bootstrap_difference(
        unit_radius_tensor[0].numpy(), unit_radius_tensor[1].numpy(), 19
    )
    rate_diff, rate_ci = _paired_bootstrap_difference(
        mean_unit_rate[0].numpy(), mean_unit_rate[1].numpy(), 20
    )
    sustained = [temporal[index]["step_sustained_index"] for index in range(2)]
    radius_relative = abs(radius_diff) / max(float(unit_radius_tensor.mean()), 1e-8)
    sustained_diff = sustained[0] - sustained[1]
    roles = {"bank_a": "bank A", "bank_b": "bank B", "status": "未形成预期功能配对"}
    if radius_relative >= 0.05 and abs(sustained_diff) >= 0.05 and radius_diff * sustained_diff < 0:
        smaller = 0 if radius_diff < 0 else 1
        more_sustained = 0 if sustained_diff > 0 else 1
        if smaller == more_sustained:
            larger = 1 - smaller
            roles = {
                "bank_a": "midget-like candidate" if smaller == 0 else "parasol-like candidate",
                "bank_b": "midget-like candidate" if smaller == 1 else "parasol-like candidate",
                "status": "候选功能配对；仅单 seed",
            }
    banks = []
    for bank in range(2):
        banks.append(
            {
                "bank": "A" if bank == 0 else "B",
                "unit_count": int(unit_radius_tensor.shape[1]),
                "sigma_degs": float(sigma[bank]),
                "effective_radius_degs": float(unit_radius_tensor[bank].mean()),
                "sustained_mix": float(mix[bank]),
                "transient_mix": float(1.0 - mix[bank]),
                "membrane_tau_ms": float(membrane_tau[bank]),
                "adaptation_tau_ms": float(adaptation_tau[bank]),
                "adaptation_gain": float(adaptation_gain[bank]),
                "rate_tau_ms": float(rate_tau[bank]),
                "amacrine_gain": float(amacrine_gain[bank]),
                "threshold": float(threshold[bank]),
                "mean_rate": float(mean_unit_rate[bank].mean()),
                "active_fraction": float((mean_unit_rate[bank] > 0).float().mean()),
                "decoder_magnitude_on": float(decoder_magnitude[bank, 0]),
                "decoder_magnitude_off": float(decoder_magnitude[bank, 1]),
                "unit_radius_degs": unit_radius_tensor[bank].tolist(),
                "unit_mean_rate": mean_unit_rate[bank].tolist(),
                **temporal[bank],
            }
        )
    return {
        "banks": banks,
        "differences": {
            "effective_radius_a_minus_b": radius_diff,
            "effective_radius_bootstrap_ci": radius_ci,
            "mean_rate_a_minus_b": rate_diff,
            "mean_rate_bootstrap_ci": rate_ci,
            "step_sustained_index_a_minus_b": sustained_diff,
        },
        "unit_feature_silhouette": separation,
        "role_assignment": roles,
    }


def _temporal_features(core: V9RetinaCore, dt_ms: float) -> list[dict[str, float]]:
    device = next(core.parameters()).device
    cone_count = core.rgc.unit_centers_degs.shape[0]
    center = cone_count // 2
    stimuli = {}
    impulse = torch.zeros(1, 320, cone_count, device=device)
    impulse[:, 224, center] = 1.0
    stimuli["impulse"] = impulse
    step = torch.zeros_like(impulse)
    step[:, 224:, center] = 1.0
    stimuli["step"] = step
    flicker = torch.zeros_like(impulse)
    flicker[:, 224:, center] = torch.arange(96, device=device).remainder(2).mul(2).sub(1)
    stimuli["flicker"] = flicker
    responses = {}
    with torch.no_grad():
        for name, stimulus in stimuli.items():
            output, _ = core.forward_sequence(stimulus)
            responses[name] = output.rates[0, :, :, 0].mean(dim=-1).cpu()
    features = []
    for bank in range(2):
        row: dict[str, float] = {}
        for name, response in responses.items():
            trace = response[:, bank]
            baseline = trace[:224].mean()
            evoked = (trace[224:] - baseline).abs()
            peak = int(evoked.argmax())
            weights = evoked / evoked.sum().clamp_min(1e-8)
            indices = torch.arange(evoked.numel(), dtype=evoked.dtype)
            center_of_mass = (indices * weights).sum()
            width = torch.sqrt(((indices - center_of_mass).square() * weights).sum())
            row[f"{name}_peak_ms"] = peak * dt_ms
            row[f"{name}_integration_width_ms"] = float(width * dt_ms)
        step_evoked = (responses["step"][:, bank][224:] - responses["step"][:, bank][:224].mean()).abs()
        row["step_sustained_index"] = float(
            step_evoked[-20:].mean() / step_evoked.max().clamp_min(1e-8)
        )
        features.append(row)
    return features


def _dynamic_rf(
    core: V9RetinaCore,
    validation: Sequence[torch.Tensor],
    *,
    context_pairs: int,
    lag_steps: int,
    dt_ms: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    pair_rows = []
    first_check: dict[str, Any] | None = None
    core.eval()
    for pair_index in range(context_pairs):
        source = torch.roll(validation[pair_index % len(validation)], pair_index * 7, dims=0).to(device)
        probe = source[-lag_steps:] * 0.65
        contexts = (source[:-lag_steps] * 0.3, source[:-lag_steps])
        states = []
        outputs = []
        probes = []
        for context in contexts:
            with torch.no_grad():
                _, context_state = core.forward_sequence(context[None])
            probe_leaf = probe.detach().clone().requires_grad_(True)
            output, _ = core.forward_sequence(probe_leaf[None], detach_v9_state(context_state))
            states.append(context_state)
            outputs.append(output)
            probes.append(probe_leaf)
        selected = (
            outputs[0].rates[:, -1] + outputs[1].rates[:, -1]
        )[0].argmax(dim=-1)
        group_rows = []
        kernels_by_context = []
        for context_index, output in enumerate(outputs):
            kernels = []
            for bank in range(2):
                for polarity in range(2):
                    unit = int(selected[bank, polarity])
                    score = output.rates[0, -1, bank, polarity, unit]
                    gradient = torch.autograd.grad(
                        score,
                        probes[context_index],
                        retain_graph=True,
                    )[0]
                    local_cones = core.rgc.support_indices[1][
                        core.rgc.support_indices[0] == unit
                    ]
                    kernels.append((bank, polarity, unit, local_cones, gradient[:, local_cones]))
            kernels_by_context.append(kernels)
        for low, high in zip(*kernels_by_context, strict=True):
            bank, polarity, unit, local_cones, low_kernel = low
            high_kernel = high[4]
            low_norm = low_kernel.norm()
            high_norm = high_kernel.norm()
            cosine_distance = 1.0 - float(
                F.cosine_similarity(low_kernel.flatten(), high_kernel.flatten(), dim=0)
            )
            low_peak, low_width = _temporal_kernel_metrics(low_kernel, dt_ms)
            high_peak, high_width = _temporal_kernel_metrics(high_kernel, dt_ms)
            low_spatial = _spatial_kernel_moment(core, unit, local_cones, low_kernel)
            high_spatial = _spatial_kernel_moment(core, unit, local_cones, high_kernel)
            group_rows.append(
                {
                    "bank": bank,
                    "polarity": polarity,
                    "unit": unit,
                    "raw_kernel_norm_ratio": float(high_norm / low_norm.clamp_min(1e-8)),
                    "gain_normalized_cosine_distance": cosine_distance,
                    "temporal_peak_shift_ms": high_peak - low_peak,
                    "temporal_integration_width_shift_ms": high_width - low_width,
                    "spatial_second_moment_shift": high_spatial - low_spatial,
                }
            )
        pair_rows.append(
            {
                key: float(np.mean([row[key] for row in group_rows]))
                for key in (
                    "raw_kernel_norm_ratio",
                    "gain_normalized_cosine_distance",
                    "temporal_peak_shift_ms",
                    "temporal_integration_width_shift_ms",
                    "spatial_second_moment_shift",
                )
            }
        )
        if first_check is None:
            first_check = _finite_difference_check(
                core,
                states[0],
                probe,
                kernels_by_context[0][0],
                seed,
            )
    summary = {}
    for key in pair_rows[0]:
        values = np.asarray([row[key] for row in pair_rows])
        summary[key] = float(values.mean())
        summary[f"{key}_bootstrap_ci"] = _bootstrap_ci(values, seed + len(summary))
    reset_effect = _reset_control(core, validation[0][-lag_steps:].to(device) * 0.65)
    recovery = _recovery_curve(core, validation, lag_steps, dt_ms, device)
    finite_difference = first_check or {"status": "not_run"}
    if finite_difference.get("relative_error", 1.0) > 0.5:
        conclusion = "not_identifiable"
    elif summary["gain_normalized_cosine_distance_bootstrap_ci"][0] > 1e-4:
        conclusion = "supported" if recovery[-1]["effect"] < recovery[0]["effect"] else "not_supported"
    elif abs(summary["raw_kernel_norm_ratio"] - 1.0) > 0.05:
        conclusion = "gain_only"
    else:
        conclusion = "not_supported"
    return {
        "status": conclusion,
        "context_pair_count": context_pairs,
        "representative_units_per_bank_polarity": 1,
        "lag_steps": lag_steps,
        "pair_metrics": pair_rows,
        "summary": summary,
        "state_reset_control_effect": reset_effect,
        "recovery_curve": recovery,
        "finite_difference_direction_check": finite_difference,
    }


def _temporal_kernel_metrics(kernel: torch.Tensor, dt_ms: float) -> tuple[float, float]:
    strength = kernel.abs().sum(dim=1)
    peak = int(strength.argmax()) * dt_ms
    weights = strength / strength.sum().clamp_min(1e-8)
    indices = torch.arange(strength.numel(), device=kernel.device, dtype=kernel.dtype)
    center = (indices * weights).sum()
    width = torch.sqrt(((indices - center).square() * weights).sum()) * dt_ms
    return peak, float(width)


def _spatial_kernel_moment(
    core: V9RetinaCore,
    unit: int,
    cones: torch.Tensor,
    kernel: torch.Tensor,
) -> float:
    strength = kernel.abs().sum(dim=0)
    positions = core.rgc.unit_centers_degs
    distance_sq = (positions[cones] - positions[unit]).square().sum(dim=1)
    return float((strength * distance_sq).sum() / strength.sum().clamp_min(1e-8))


def _finite_difference_check(
    core: V9RetinaCore,
    state: Any,
    probe: torch.Tensor,
    kernel_entry: tuple[int, int, int, torch.Tensor, torch.Tensor],
    seed: int,
) -> dict[str, float | str]:
    bank, polarity, unit, local_cones, kernel = kernel_entry
    local_cones = local_cones[:8]
    generator = torch.Generator(device=probe.device).manual_seed(seed)
    direction = torch.randn(
        (probe.shape[0], local_cones.numel()),
        generator=generator,
        device=probe.device,
    )
    direction = direction / direction.norm().clamp_min(1e-8)
    autodiff = float((kernel[:, : local_cones.numel()] * direction).sum())
    epsilon = 1e-3
    values = []
    for sign in (-1.0, 1.0):
        shifted = probe.clone()
        shifted[:, local_cones] += sign * epsilon * direction
        with torch.no_grad():
            output, _ = core.forward_sequence(shifted[None], detach_v9_state(state))
        values.append(float(output.rates[0, -1, bank, polarity, unit]))
    finite_difference = (values[1] - values[0]) / (2.0 * epsilon)
    relative_error = abs(autodiff - finite_difference) / (
        abs(autodiff) + abs(finite_difference) + 1e-8
    )
    return {
        "status": "ok" if relative_error <= 0.5 else "mismatch",
        "autodiff_directional_derivative": autodiff,
        "finite_difference_directional_derivative": finite_difference,
        "relative_error": relative_error,
        "unit_count": 1,
        "local_cone_count": int(local_cones.numel()),
    }


def _reset_control(core: V9RetinaCore, probe: torch.Tensor) -> float:
    with torch.no_grad():
        first, _ = core.forward_sequence(probe[None])
        second, _ = core.forward_sequence(probe[None])
    return float((first.rates - second.rates).abs().mean())


def _recovery_curve(
    core: V9RetinaCore,
    validation: Sequence[torch.Tensor],
    lag_steps: int,
    dt_ms: float,
    device: torch.device,
) -> list[dict[str, float]]:
    delays_ms = (0, 100, 200, 300, 500)
    effects = []
    for delay_ms in delays_ms:
        rows = []
        delay_steps = round(delay_ms / dt_ms)
        for source in validation[:4]:
            source = source.to(device)
            with torch.no_grad():
                _, low = core.forward_sequence((source[:160] * 0.3)[None])
                _, high = core.forward_sequence(source[:160][None])
                if delay_steps:
                    neutral = torch.zeros(1, delay_steps, source.shape[1], device=device)
                    _, low = core.forward_sequence(neutral, low)
                    _, high = core.forward_sequence(neutral, high)
                probe = (source[-lag_steps:] * 0.65)[None]
                low_output, _ = core.forward_sequence(probe, low)
                high_output, _ = core.forward_sequence(probe, high)
            rows.append(float((high_output.rates - low_output.rates).abs().mean()))
        effects.append({"delay_ms": float(delay_ms), "effect": float(np.mean(rows))})
    return effects


def _parameter_bounds(trainer: V9Trainer) -> dict[str, Any]:
    encoder = trainer.core.rgc
    rows = []
    for name, (lower, upper, _) in encoder._PARAMETERS.items():
        values = encoder.bounded(name).detach().cpu()
        for bank, value in enumerate(values):
            fraction = (float(value) - lower) / (upper - lower)
            rows.append(
                {
                    "name": name,
                    "bank": "A" if bank == 0 else "B",
                    "value": float(value),
                    "lower": lower,
                    "upper": upper,
                    "fraction": fraction,
                    "near_boundary": fraction < 0.02 or fraction > 0.98,
                }
            )
    return {
        "parameters": rows,
        "near_boundary_count": sum(row["near_boundary"] for row in rows),
    }


def _silhouette(features: torch.Tensor) -> float:
    flat = features.reshape(-1, features.shape[-1]).float()
    flat = (flat - flat.mean(dim=0)) / flat.std(dim=0).clamp_min(1e-6)
    labels = torch.arange(2).repeat_interleave(features.shape[1])
    distances = torch.cdist(flat, flat)
    values = []
    for index, label in enumerate(labels):
        same = labels == label
        same[index] = False
        other = labels != label
        within = distances[index, same].mean()
        between = distances[index, other].mean()
        values.append((between - within) / torch.maximum(within, between).clamp_min(1e-8))
    return float(torch.stack(values).mean())


def _paired_bootstrap_difference(
    first: np.ndarray, second: np.ndarray, seed: int
) -> tuple[float, list[float]]:
    differences = np.asarray(first) - np.asarray(second)
    return float(differences.mean()), _bootstrap_ci(differences, seed)


def _bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(1000, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def write_v9_artifacts(
    run_dir: Path,
    metrics: dict[str, Any],
    training_rows: Sequence[dict[str, Any]],
    report_context: dict[str, Any],
) -> Path:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _write_plots(plots_dir, metrics, training_rows)
    metrics_path = run_dir / "final_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_metrics_csv(run_dir / "evaluation_metrics.csv", metrics)
    report_path = run_dir / "final_report_zh.md"
    report_path.write_text(_report_text(metrics, report_context), encoding="utf-8")
    return report_path


def _write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    rows = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(f"{prefix}.{key}" if prefix else key, item)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            rows.append((prefix, value))

    visit("", metrics)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        writer.writerows(rows)


def _write_plots(
    plots_dir: Path,
    metrics: dict[str, Any],
    training_rows: Sequence[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in training_rows]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(steps, [row["raw_reconstruction"] for row in training_rows])
    axes[0].set_ylabel("current MSE")
    axes[1].plot(steps, [row["energy"] for row in training_rows], label="energy")
    axes[1].plot(steps, [row.get("energy_budget", np.nan) for row in training_rows], label="budget")
    axes[1].set_xlabel("optimizer step")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    banks = metrics["bank_features"]["banks"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    labels = ["A", "B"]
    axes[0].bar(labels, [bank["effective_radius_degs"] for bank in banks])
    axes[0].set_title("effective radius")
    axes[1].bar(labels, [bank["step_sustained_index"] for bank in banks])
    axes[1].set_title("sustained index")
    axes[2].bar(labels, [bank["mean_rate"] for bank in banks])
    axes[2].set_title("mean rate")
    fig.tight_layout()
    fig.savefig(plots_dir / "bank_feature_summary.png", dpi=160)
    plt.close(fig)

    pairs = metrics["dynamic_rf"]["pair_metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].hist([row["raw_kernel_norm_ratio"] for row in pairs])
    axes[0].set_title("kernel norm ratio")
    axes[1].hist([row["gain_normalized_cosine_distance"] for row in pairs])
    axes[1].set_title("cosine distance")
    fig.tight_layout()
    fig.savefig(plots_dir / "dynamic_rf_context_summary.png", dpi=160)
    plt.close(fig)

    recovery = metrics["dynamic_rf"]["recovery_curve"]
    fig, axis = plt.subplots(figsize=(5, 3.5))
    axis.plot([row["delay_ms"] for row in recovery], [row["effect"] for row in recovery], marker="o")
    axis.set_xlabel("recovery delay (ms)")
    axis.set_ylabel("context effect")
    fig.tight_layout()
    fig.savefig(plots_dir / "dynamic_rf_recovery.png", dpi=160)
    plt.close(fig)


def _report_text(metrics: dict[str, Any], context: dict[str, Any]) -> str:
    reconstruction = metrics["reconstruction"]
    roles = metrics["bank_features"]["role_assignment"]
    dynamic = metrics["dynamic_rf"]
    representation = "GO" if reconstruction["skill_score"] > 0 else "NO-GO"
    emergence = "GO" if "候选功能配对" in roles["status"] else "NO-GO"
    dynamic_status = "GO" if dynamic["status"] == "supported" else (
        "PARTIAL" if dynamic["status"] == "gain_only" else "NO-GO"
    )
    paths = context["artifact_paths"]
    sections = [
        ("1. 执行状态和一句话结论", f"状态：{context['status']}。current representation={representation}，bank emergence={emergence}，dynamic RF={dynamic_status}。"),
        ("2. 研究目标及允许的结论层级", "在预设 RGC 输出、ON/OFF 极性和两个匿名对称 bank 下，检验功能身份是否由训练形成，并检验 preceding context 是否通过内部状态改变 RGC-level effective RF。本轮只允许声称单 seed 的机制方向证据。"),
        ("3. 原 v8 问题与本次修改映射", "移除了 population-specific support、mix bounds、decoder warmup、decoder spatial logits、无条件二次 energy cost、短 t_bptt 和每轮完整 train_eval。改为对称匿名 bank、tied decoder、256-step 可微窗口和 inequality energy budget。"),
        ("4. 实际修改文件、关键类/函数和 diff 摘要", context["changed_files"]),
        ("5. 数据审计、实际启用的数据模式和限制", json.dumps(context["data_manifest"], ensure_ascii=False, indent=2)),
        ("6. 最终 resolved config", json.dumps(context["resolved_config"], ensure_ascii=False, indent=2)),
        ("7. BPTT 实现和 256-vs-320 梯度审计", json.dumps(context["gradient_audit"], ensure_ascii=False, indent=2)),
        ("8. smoke test 与单元测试结果", json.dumps(context["test_results"], ensure_ascii=False, indent=2)),
        ("9. 完整训练命令、环境、GPU、runtime 和 peak memory", json.dumps(context["runtime"], ensure_ascii=False, indent=2)),
        ("10. 训练曲线和稳定性", f"见 `plots/training_curves.png`；NaN/Inf={context['nan_inf']}。"),
        ("11. checkpoint 选择", f"主 checkpoint：`{context['checkpoint']}`。"),
        ("12. reconstruction 与 energy 结果", json.dumps({"reconstruction": reconstruction, "resources": metrics["resources"]}, ensure_ascii=False, indent=2)),
        ("13. anonymous bank 功能分化结果", json.dumps(metrics["bank_features"], ensure_ascii=False, indent=2)),
        ("14. dynamic RF matched-context、state reset 和 recovery 结果", json.dumps(dynamic, ensure_ascii=False, indent=2)),
        ("15. 参数边界与失败诊断", json.dumps(metrics["parameter_bounds"], ensure_ascii=False, indent=2)),
        ("16. 已知事实、合理推断、仍需验证内容", "已知事实来自落盘指标；bank 身份和 dynamic RF 解释仅按预注册判据生成。单 seed、12/4 个 source、synthetic noise 和 4° 偏心度限制外推。"),
        ("17. GO / PARTIAL / NO-GO", f"- current representation：{representation}\n- midget-like/parasol-like 功能配对：{emergence}\n- state-dependent dynamic RF：{dynamic_status}（原始状态 `{dynamic['status']}`）"),
        ("18. 下一步（最多三项）", "1. 仅在本轮机制方向为 GO/PARTIAL 时增加独立 seed。\n2. 用更大 source-disjoint 自然序列复核不确定性。\n3. finite-difference 不一致时改进可辨识 readout，而不是增加细胞机制。"),
        ("19. 所有产物的绝对路径", json.dumps(paths, ensure_ascii=False, indent=2)),
    ]
    body = "\n\n".join(f"## {title}\n\n{content}" for title, content in sections)
    return f"# Retina SNN v9 匿名双通路与动态 RF 执行报告\n\n{body}\n"
