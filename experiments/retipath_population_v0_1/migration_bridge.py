from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from experiments.retipath_multiscale_v0.circuit import (
    LocalRetipathV0, _delay1, _lowpass, area_integral_weights,
)
from experiments.retipath_multiscale_v0.contracts import (
    DT_MS, Intervention as G1Intervention, InterventionSpec, StimulusBatch, regular_pixel_bounds,
)
from experiments.retipath_multiscale_v0.test_contract import (
    event_fixture, fixture as g1_fixture, gaussian_pixel_average,
)
from models.mechanistic_retina.retipath_spatial_ei import exponential_sequence
from models.mechanistic_retina.state import fixed_one_bin_history_state
from .circuit import BCOutput, Intervention, PopulationRetipath, Stimulus


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "output/evaluations/retipath_population_stage_a5_20260918"
TOL = 2e-12
PROTOCOL = {
    "task": "Population Stage A.5 deterministic migration bridge, not a benchmark",
    "models": ["G1_default", "Population_LegacyPReLU", "Population_BaselineSoftplus"],
    "parameters": "unchanged constructor defaults; no seeds, fitting, optimizer or checkpoints",
    "dtype": "float64", "device": "cpu", "threads": 1, "dt_ms": DT_MS,
    "matched_fixture": "G1 test_contract Gaussian sigma=.30 deg, 32 grid, original six-bin profile, 24 bins",
    "events": "G1 event_fixture repeated identically across the two Population RGCs",
    "grid": {"sizes": [32, 64], "square_half_width_deg": .25,
             "gaussian_sigma_deg": .30, "scale_sigmas_deg": [.15, .60]},
    "f2": {"spatial_fixtures": ["G1_FINE_STRIPES_CR", "MIRROR_NULL_CR"], "grid": 64,
           "frequency_hz": 4., "contrast": .5, "total_bins": 1200, "measurement": [900, 1200],
           "events": "zero", "phase_time": "n/150 seconds; stimulus metadata uses bin midpoint",
           "stripes": "exact alternating 64-column pattern from G1 test_02, width=1/32 degree",
           "mirror_null": "sign(x), symmetric physical half-fields, not a prior G1 F2 protocol",
           "provenance": "G1 had null-sum stripes but no dedicated F2 experiment; sinusoidal time extension is new requested bridge fixture",
           "fourier": "C_h=2/N sum((y-mean(y))*exp(-i*2*pi*4*h*n/150)); component first, then RMS",
           "floor": "4096*float64_eps*max_abs(measurement trace); numerical reference only",
           "grouping": "G1 sustained/transient remain separate; Population ON/OFF remain separate",
           "selection": "no activation/phase/parameter/window selection from responses"},
    "checks": {"absolute_tolerance": TOL, "physical_scale_min_delta_probability": 1e-8,
               "scale_tolerance_source": "existing G1 test_02",
               "future_cut": 8, "delta_logit_sign": "block minus normal"},
    "g1_missing_block_api": "H1/AC evaluation uses fixed algebra replay; normal/direct replay validated against native G1",
    "contribution": "d_E/d_I are drive contributions; logit is not an additive decomposition",
    "stop": "no Stage B, no architecture edits, no new trainable parameters",
}


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))}


def old_ports(trace) -> dict[str, Tensor]:
    i, s, o = trace.inputs, trace.states, trace.outputs
    return {"q": i["x"][..., 0], "u_H": i["u_H"][..., 0], "h_H": s["h"][..., 0],
            "H1_feedback": o["f"][..., 0], "u_B": i["u_B"][..., 0],
            "s_B": s["s_B"], "r_B": o["o_B"], "delta_r_B": o["o_B"],
            "u_A": i["u_A"], "a_A": s["a_AC"], "o_A": s["a_AC"], "delta_o_A": s["a_AC"],
            "c_E": o["d_E"].transpose(-1, -2).unsqueeze(2),
            "c_I": -o["j_I"].transpose(-1, -2).unsqueeze(2),
            "d_E": o["uE"], "d_I": o["uI"], "gE": o["gE"], "gI": o["gI"],
            "V": s["V"], "adaptation": s["z"], "history": s["q"],
            "logit": o["ell"], "probability": o["p"]}


def old_replay(model: LocalRetipathV0, stimulus: StimulusBatch, events: Tensor,
               intervention: Intervention) -> dict[str, Tensor]:
    """Evaluation-only G1 algebra; zero the requested edge without mutating its parameters."""
    p = model.physical_parameters()
    q = stimulus.values[..., 0] @ area_integral_weights(stimulus.pixel_bounds, model.input_xy).T
    uh = q @ model.G.T
    h = _lowpass(_delay1(uh), p["tau_h"])
    feedback = p["a_h"] * (h @ model.G)
    if intervention is Intervention.BLOCK_H1_FEEDBACK:
        feedback = torch.zeros_like(feedback)
    ub = q - feedback
    fast, slow = _lowpass(ub, p["tau_f"]), _lowpass(ub, p["tau_s"])
    state = torch.stack((fast, fast - slow), -1)
    output = torch.where(state >= 0, state, p["alpha"] * state)
    we = torch.stack((p["delta_e"], p["delta_e"].new_zeros(()))).softmax(0)
    wi = torch.stack((p["delta_i"], p["delta_i"].new_zeros(()))).softmax(0)
    ce = p["g_e"] * (output * we).sum(-1)[..., None] * model.D.T
    if intervention is Intervention.BLOCK_DIRECT_BC_DRIVE:
        ce = torch.zeros_like(ce)
    de = ce.sum(-2, keepdim=True)
    ua = torch.einsum("ji,btid->btjd", model.A, output)
    a = _lowpass(_delay1(ua), torch.stack((p["tau_a1"], p["tau_a2"])))
    signed_i = -p["g_i"] * (a * wi).sum(-1)[..., None] * model.I.T
    if intervention is Intervention.BLOCK_AC_POSTSYNAPTIC_DRIVE:
        signed_i = torch.zeros_like(signed_i)
    di = -signed_i.sum(-2, keepdim=True)
    b0 = math.log(math.expm1(1.))
    ge, gi = F.softplus(b0 + de), F.softplus(b0 + di)
    total = 1 + ge + gi
    voltage = exponential_sequence((ge - gi / 3) / total, -torch.expm1(-DT_MS * total / 60), 2 / 9)
    membrane = voltage.mean(-1) - 2 / 9
    adaptation = _lowpass(membrane, 120.)
    history = fixed_one_bin_history_state(events, math.exp(-DT_MS / 40))
    logit = 8 * membrane - .5 * adaptation - history + p["bias"]
    return {"q": q, "u_H": uh, "h_H": h, "H1_feedback": feedback, "u_B": ub,
            "s_B": state, "r_B": output, "delta_r_B": output, "u_A": ua,
            "a_A": a, "o_A": a, "delta_o_A": a, "c_E": ce.transpose(-1, -2).unsqueeze(2),
            "c_I": -signed_i.transpose(-1, -2).unsqueeze(2), "d_E": de, "d_I": di,
            "gE": ge, "gI": gi, "V": voltage, "adaptation": adaptation, "history": history,
            "logit": logit, "probability": logit.sigmoid()}


def run_model(model, stimulus: StimulusBatch, events: Tensor,
              intervention: Intervention = Intervention.NORMAL) -> dict[str, Tensor]:
    if isinstance(model, LocalRetipathV0):
        if intervention in (Intervention.NORMAL, Intervention.BLOCK_DIRECT_BC_DRIVE):
            return old_ports(model(stimulus, observed_events=events,
                                   intervention=InterventionSpec(G1Intervention(intervention.value))))
        return old_replay(model, stimulus, events, intervention)
    pop_input = Stimulus(stimulus.values[..., 0], stimulus.pixel_bounds, stimulus.pixel_area,
                        stimulus.time_ms, stimulus.input_valid[..., 0])
    trace = model(pop_input, observed_events=events.repeat(1, 1, 2), intervention=intervention)
    fields = {**trace.inputs, **trace.states, **trace.outputs}
    return {key: fields[key] for key in (
        "q", "u_H", "h_H", "H1_feedback", "u_B", "s_B", "r_B", "delta_r_B", "u_A",
        "a_A", "o_A", "delta_o_A", "c_E", "c_I", "d_E", "d_I", "gE", "gI", "V",
        "adaptation", "history", "logit", "probability")}


def save_trace(path: Path, trace: dict[str, Tensor]) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **{name: value[0].numpy() for name, value in trace.items()})


def scalar(value: Tensor) -> float:
    return float(value.abs().max())


def rms(value: Tensor) -> float:
    return float(value.square().mean().sqrt())


def harmonic_groups(model_name: str, port: str, values: np.ndarray) -> dict[str, np.ndarray]:
    if port in ("u_B", "s_B", "r_B", "delta_r_B"):
        if model_name == "G1_default" and values.ndim == 3:
            return {"sustained": values[..., 0], "transient": values[..., 1]}
        if model_name != "G1_default":
            return {"ON": values[:, :25], "OFF": values[:, 25:]}
    return {"all": values.reshape(values.shape[0], -1)}


def harmonics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = values[900:1200].reshape(300, -1)
    centered = signal - signal.mean(0)
    n = np.arange(900, 1200)
    coefficients = np.stack([2 / 300 * np.exp(-2j * np.pi * 4 * h * n / 150) @ centered
                             for h in (1, 2)])
    floor = 4096 * np.finfo(np.float64).eps * np.abs(signal).max(0)
    return coefficients, np.abs(coefficients), floor


def harmonic_rows(model: str, fixture_name: str, port: str, values: np.ndarray) -> list[dict]:
    rows = []
    for group, trace in harmonic_groups(model, port, values).items():
        trace = trace.reshape(1200, -1)
        coeff, amp, floor = harmonics(trace)
        for component in range(trace.shape[1]):
            c, a = coeff[:, component], amp[:, component]
            rows.append({"model": model, "fixture": fixture_name, "port": port, "group": group,
                         "component": str(component), "C1_real": float(c[0].real), "C1_imag": float(c[0].imag),
                         "C2_real": float(c[1].real), "C2_imag": float(c[1].imag),
                         "F1": float(a[0]), "F2": float(a[1]), "floor": float(floor[component]),
                         "phase1_rad": float(np.angle(c[0])) if a[0] > floor[component] else None,
                         "phase2_rad": float(np.angle(c[1])) if a[1] > floor[component] else None})
        rows.append({"model": model, "fixture": fixture_name, "port": port, "group": group,
                     "component": "RMS", "C1_real": None, "C1_imag": None, "C2_real": None, "C2_imag": None,
                     "F1": float(np.sqrt(np.mean(amp[0] ** 2))), "F2": float(np.sqrt(np.mean(amp[1] ** 2))),
                     "floor": float(np.sqrt(np.mean(floor ** 2))), "phase1_rad": None, "phase2_rad": None})
        c, a, fl = harmonics(trace.mean(1, keepdims=True))
        rows.append({"model": model, "fixture": fixture_name, "port": port, "group": group,
                     "component": "SPATIAL_MEAN", "C1_real": float(c[0, 0].real), "C1_imag": float(c[0, 0].imag),
                     "C2_real": float(c[1, 0].real), "C2_imag": float(c[1, 0].imag),
                     "F1": float(a[0, 0]), "F2": float(a[1, 0]), "floor": float(fl[0]),
                     "phase1_rad": float(np.angle(c[0, 0])) if a[0, 0] > fl[0] else None,
                     "phase2_rad": float(np.angle(c[1, 0])) if a[1, 0] > fl[0] else None})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=False)
    protected = [ROOT / path for path in (
        "AGENTS.md", "docs/RETIPATH_POPULATION_ARCHITECTURE_V0_1.md",
        "docs/RETIPATH_POPULATION_STAGE_A.md", "docs/RETIPATH_G2_MULTISCALE_RESULTS.md",
        "docs/RETIPATH_G3_ROBUSTNESS_RESULTS.md", "docs/RETIPATH_F2_LOCALIZATION_PILOT.md",
        "models/mechanistic_retina/state.py", "models/mechanistic_retina/retipath_spatial_ei.py",
        "evaluation/mechanistic_retina/mechanism_observation.py")]
    protected += list((ROOT / "experiments/retipath_multiscale_v0").glob("*.py"))
    protected += [ROOT / "experiments/retipath_multiscale_v0/protocol.json"]
    protected += [ROOT / "experiments/retipath_population_v0_1" / name
                  for name in ("__init__.py", "circuit.py", "test_stage_a.py")]
    before = hashes(protected)
    write_json(DESTINATION / "protocol.json", PROTOCOL)
    write_json(DESTINATION / "SOURCE_LOCK.json", {
        "utc_before_forward": datetime.now(timezone.utc).isoformat(), "protected": before,
        "evaluator": hashes([Path(__file__).resolve()]),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}})
    torch.set_num_threads(1)
    models = {"G1_default": LocalRetipathV0(dtype=torch.float64),
              "Population_LegacyPReLU": PopulationRetipath(),
              "Population_BaselineSoftplus": PopulationRetipath(bc_output=BCOutput.BASELINE_SOFTPLUS)}
    original_states = {name: {key: value.clone() for key, value in model.state_dict().items()}
                       for name, model in models.items()}
    write_json(DESTINATION / "DEFAULT_PARAMETERS.json", {
        name: {"parameter_count": sum(p.numel() for p in model.parameters()),
               "physical_values": {key: value.detach().tolist() for key, value in model.physical_parameters().items()}}
        for name, model in models.items()})
    bounds = regular_pixel_bounds(32, dtype=torch.float64)
    matched = g1_fixture(gaussian_pixel_average(bounds, .30), bounds)
    events = event_fixture(matched)
    with (DESTINATION / "matched_input.npz").open("xb") as stream:
        np.savez_compressed(stream, values=matched.values.numpy(), pixel_bounds=bounds.numpy(),
                            time_ms=matched.time_ms.numpy(), observed_events=events.numpy())
    checks, intervention_rows, trajectory_rows, harmonic_table, f2_summary = [], [], [], [], []

    def check(model: str, item: str, metric: str, value: float, limit: float = TOL,
              relation: str = "<=") -> None:
        checks.append({"model": model, "item": item, "metric": metric, "value": value,
                       "relation": relation, "limit": limit,
                       "passed": bool(math.isfinite(value) and (value <= limit if relation == "<=" else value > limit))})

    with torch.inference_mode():
        for name, model in models.items():
            print(f"Checking frozen graph: {name}", flush=True)
            traces = {kind: run_model(model, matched, events, kind) for kind in Intervention}
            normal = traces[Intervention.NORMAL]
            for kind, trace in traces.items():
                save_trace(DESTINATION / f"matched__{name}__{kind.value}.npz", trace)
                check(name, "finite", kind.value, float(not all(bool(torch.isfinite(v).all()) for v in trace.values())), 0.)
                check(name, "conductance", f"{kind.value}_min_gE", float(trace["gE"].min()), 0., ">")
                check(name, "conductance", f"{kind.value}_min_gI", float(trace["gI"].min()), 0., ">")
                p = model.physical_parameters()
                bias = p["bias"]
                v = torch.full_like(trace["gE"][:, 0], 2 / 9)
                a = torch.zeros_like(v.mean(-1))
                expected_v, expected_logit = [], []
                for t in range(24):
                    ge, gi = trace["gE"][:, t], trace["gI"][:, t]
                    total = 1 + ge + gi
                    target = (ge - gi / 3) / total
                    v = target + (v - target) * torch.exp(-DT_MS * total / 60)
                    a = math.exp(-DT_MS / 120) * a + (1 - math.exp(-DT_MS / 120)) * (v.mean(-1) - 2 / 9)
                    expected_v.append(v)
                    expected_logit.append(8 * (v.mean(-1) - 2 / 9) - .5 * a - trace["history"][:, t] + bias)
                check(name, "conductance", f"{kind.value}_voltage_recurrence", scalar(trace["V"] - torch.stack(expected_v, 1)))
                check(name, "conductance", f"{kind.value}_logit_recurrence", scalar(trace["logit"] - torch.stack(expected_logit, 1)))
                if kind is Intervention.NORMAL:
                    continue
                keep = ["h_H", "history"]
                if kind is Intervention.BLOCK_H1_FEEDBACK:
                    zero_ports = ["H1_feedback"]
                else:
                    keep += ["s_B", "delta_r_B", "u_A", "a_A", "o_A", "H1_feedback"]
                    keep += ["d_I", "gI"] if kind is Intervention.BLOCK_DIRECT_BC_DRIVE else ["d_E", "gE"]
                    zero_ports = ["c_E", "d_E"] if kind is Intervention.BLOCK_DIRECT_BC_DRIVE else ["c_I", "d_I"]
                    tonic = "gE" if kind is Intervention.BLOCK_DIRECT_BC_DRIVE else "gI"
                    check(name, "blocks", f"{kind.value}_tonic", scalar(trace[tonic] - 1))
                for port in keep:
                    check(name, "blocks", f"{kind.value}_preserve_{port}", scalar(trace[port] - normal[port]))
                for port in zero_ports:
                    check(name, "blocks", f"{kind.value}_zero_{port}", scalar(trace[port]))
                check(name, "blocks", f"{kind.value}_recomputed_logit", scalar(trace["logit"] - normal["logit"]), 1e-12, ">")
                for r in range(normal["logit"].shape[-1]):
                    delta = trace["logit"][0, :, r] - normal["logit"][0, :, r]
                    intervention_rows.append({"model": name, "fixture": "matched_Gaussian", "block": kind.value, "rgc": r,
                        "normal_direct_rms": rms(normal["d_E"][0, :, r]), "block_direct_rms": rms(trace["d_E"][0, :, r]),
                        "normal_ac_rms": rms(normal["d_I"][0, :, r]), "block_ac_rms": rms(trace["d_I"][0, :, r]),
                        "delta_direct_rms": rms(trace["d_E"][0, :, r] - normal["d_E"][0, :, r]),
                        "delta_ac_rms": rms(trace["d_I"][0, :, r] - normal["d_I"][0, :, r]),
                        "normal_logit_mean": float(normal["logit"][0, :, r].mean()),
                        "block_logit_mean": float(trace["logit"][0, :, r].mean()), "delta_logit_mean": float(delta.mean()),
                        "delta_logit_rms": rms(delta), "delta_logit_min": float(delta.min()), "delta_logit_max": float(delta.max())})
                    for t in range(24):
                        row = {"model": name, "fixture": "matched_Gaussian", "block": kind.value, "rgc": r,
                               "bin": t, "time_ms": float(matched.time_ms[t]),
                               "normal_logit": float(normal["logit"][0, t, r]),
                               "block_logit": float(trace["logit"][0, t, r]), "delta_logit": float(delta[t])}
                        for port in ("d_E", "d_I", "gE", "gI", "V"):
                            for k in range(2):
                                row[f"normal_{port}_{k}"] = float(normal[port][0, t, r, k])
                                row[f"block_{port}_{k}"] = float(trace[port][0, t, r, k])
                        trajectory_rows.append(row)
            if name == "G1_default":
                for kind in (Intervention.NORMAL, Intervention.BLOCK_DIRECT_BC_DRIVE):
                    replay = old_replay(model, matched, events, kind)
                    for port in normal:
                        check(name, "g1_replay", f"{kind.value}_{port}", scalar(replay[port] - traces[kind][port]))
            else:
                p = model.physical_parameters()
                expected_e = normal["delta_r_B"][:, :, None, None] * p["gamma_BR"][..., None] * model.pi_BR
                expected_a = (normal["delta_r_B"] @ model.pi_BA.T) * p["gamma_BA"]
                expected_i = normal["delta_o_A"][:, :, None, None] * p["gamma_AR"][..., model.ac_family] * model.pi_AR
                check(name, "split", "same_delta_to_direct", scalar(expected_e - normal["c_E"]))
                check(name, "split", "same_delta_to_AC", scalar(expected_a - normal["u_A"]))
                check(name, "split", "AC_to_inhibitory", scalar(expected_i - normal["c_I"]))
            square_traces = []
            gaussian_errors = []
            for size in (32, 64):
                pixels = regular_pixel_bounds(size, dtype=torch.float64)
                square = ((pixels.mean(-1).abs() < .25).all(-1)).double()
                square_traces.append(run_model(model, g1_fixture(square, pixels), events))
                q = area_integral_weights(pixels, model.input_xy) @ gaussian_pixel_average(pixels, .30)
                apertures = torch.stack((model.input_xy - .05, model.input_xy + .05), -1)
                gaussian_errors.append(scalar(q - gaussian_pixel_average(apertures, .30)))
            for port in normal:
                check(name, "Q", f"aligned_grid32_64_{port}", scalar(square_traces[0][port] - square_traces[1][port]))
            check(name, "Q", "smooth_grid64_error_minus_grid32_error", gaussian_errors[1] - gaussian_errors[0], 0.)
            checks.append({"model": name, "item": "Q_raw", "metric": "smooth_Q_error_32", "value": gaussian_errors[0], "relation": "record", "limit": None, "passed": True})
            checks.append({"model": name, "item": "Q_raw", "metric": "smooth_Q_error_64", "value": gaussian_errors[1], "relation": "record", "limit": None, "passed": True})
            pixels = regular_pixel_bounds(64, dtype=torch.float64)
            scale_traces = [run_model(model, g1_fixture(gaussian_pixel_average(pixels, scale), pixels), events)
                            for scale in (.15, .60)]
            for port in ("q", "delta_r_B", "logit", "probability"):
                check(name, "physical_scale", f"sigma_015_vs_060_{port}", scalar(scale_traces[0][port] - scale_traces[1][port]), 1e-8, ">")
            for index, scale in enumerate(("015", "060")):
                save_trace(DESTINATION / f"scale_{scale}__{name}.npz", scale_traces[index])
            future_x = matched.values.clone()
            future_x[:, 8:] *= -3
            future_e = events.clone()
            future_e[:, 8:] = 1 - future_e[:, 8:]
            future = run_model(model, replace(matched, values=future_x), future_e)
            for port in normal:
                check(name, "causality", f"future_prefix_{port}", scalar(normal[port][:, :8] - future[port][:, :8]))
            event_only = run_model(model, matched, future_e)
            check(name, "causality", "current_event_excluded", scalar(normal["logit"][:, :9] - event_only["logit"][:, :9]))
            check(name, "causality", "past_events_have_effect", scalar(normal["logit"][:, 9:] - event_only["logit"][:, 9:]), 1e-12, ">")
            expected_history = torch.zeros_like(normal["history"][:, 0])
            history_values = []
            for t in range(24):
                if t:
                    expected_history = math.exp(-DT_MS / 40) * expected_history + (1 - math.exp(-DT_MS / 40)) * events[:, t - 1]
                history_values.append(expected_history)
            check(name, "causality", "strictly_past_history_recurrence", scalar(normal["history"] - torch.stack(history_values, 1)))

        pixels = regular_pixel_bounds(64, dtype=torch.float64)
        patterns = {
            "G1_FINE_STRIPES_CR": torch.where(torch.arange(64) % 2 == 0, 1., -1.).double().expand(64, 64).flatten(),
            "MIRROR_NULL_CR": pixels.mean(-1)[:, 0].sign(),
        }
        wave = .5 * torch.sin(2 * math.pi * 4 * torch.arange(1200, dtype=torch.float64) / 150)
        with (DESTINATION / "harmonic_inputs_factorized.npz").open("xb") as stream:
            np.savez_compressed(stream, pixel_bounds=pixels.numpy(), wave=wave.numpy(),
                                **{key: value.numpy() for key, value in patterns.items()})
        n = np.arange(1200)
        c, _, _ = harmonics((2 + .3 * np.cos(2 * np.pi * 4 * n / 150) + .4 * np.sin(2 * np.pi * 8 * n / 150))[:, None])
        check("evaluator", "Fourier", "analytic_complex_coefficients", float(np.max(np.abs(c[:, 0] - np.array([.3, -.4j])))))
        ordered = ("q", "u_H", "h_H", "H1_feedback", "u_B", "s_B", "delta_r_B")
        for fixture_name, pattern in patterns.items():
            stimulus = g1_fixture(pattern, pixels, time_count=1200)
            stimulus = replace(stimulus, values=(wave[:, None] * pattern)[None, ..., None])
            zero_events = torch.zeros(1, 1200, 1, dtype=torch.float64)
            for name, model in models.items():
                print(f"Measuring fixed harmonics: {name} / {fixture_name}", flush=True)
                trace = run_model(model, stimulus, zero_events)
                save_trace(DESTINATION / f"harmonic__{name}__{fixture_name}.npz", trace)
                rows = []
                for port in (*ordered, "r_B", "a_A", "delta_o_A", "d_E", "d_I", "logit"):
                    rows += harmonic_rows(name, fixture_name, port, trace[port][0].numpy())
                harmonic_table.extend(rows)
                detectable = [port for port in ordered if any(row["F2"] > row["floor"]
                    for row in rows if row["port"] == port and row["component"] == "RMS")]
                f2_summary.append({"model": name, "fixture": fixture_name,
                    "first_resolved_F2_port": detectable[0] if detectable else None,
                    "input_spatial_mean_max_abs": scalar(stimulus.values.mean(2)),
                    "q_spatial_mean_max_abs": scalar(trace["q"].mean(-1)),
                    "BC_groups": [row for row in rows if row["port"] in ("u_B", "s_B", "delta_r_B")
                                  and row["component"] in ("RMS", "SPATIAL_MEAN")]})

    for name, model in models.items():
        unchanged = all(torch.equal(value, model.state_dict()[key]) for key, value in original_states[name].items())
        check(name, "preservation", "in_memory_parameter_or_buffer_changed", float(not unchanged), 0.)
    after = hashes(protected)
    write_json(DESTINATION / "SOURCE_AFTER.json", {"protected": after, "unchanged": before == after})
    write_csv(DESTINATION / "checks.csv", checks)
    write_csv(DESTINATION / "interventions.csv", intervention_rows)
    write_csv(DESTINATION / "matched_delta_logit.csv", trajectory_rows)
    write_csv(DESTINATION / "harmonics.csv", harmonic_table)
    write_json(DESTINATION / "summary.json", {
        "checks_total": len(checks), "checks_failed": [row for row in checks if not row["passed"]],
        "protected_sources_unchanged": before == after, "f2": f2_summary,
        "interventions": intervention_rows, "training_updates": 0,
        "first_F2_is_descriptive": True, "Stage_B_started": False})
    write_json(DESTINATION / "FILE_MANIFEST.json", hashes(list(DESTINATION.iterdir())))
    print(json.dumps({"output": str(DESTINATION), "checks": len(checks),
                      "failed": sum(not row["passed"] for row in checks), "sources_unchanged": before == after}, indent=2), flush=True)


if __name__ == "__main__":
    main()
