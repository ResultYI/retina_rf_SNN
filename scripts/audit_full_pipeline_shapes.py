from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from configs.physiology_profiles import dt_ms_from_time_axis_seconds
from data.cone_response import load_cone_response
from data.dataset import fit_log_cone_stats
from datasets.isetbio_h5_dataset import (
    ConeNormalizationStats,
    ISETBioH5Dataset,
    ISETBioH5DatasetConfig,
    collate_isetbio_h5_batch,
)
from models.cells.rgc import RGCOutput, RGCPopulationTensors
from models.retina_snn import RetinaStepDiagnostics
from training.stage1 import MidgetSamplingMode, Stage1BuildConfig, build_stage1_components


DEFAULT_H5 = ROOT / "data" / "isetbio_local_smoke" / "input_seed7.h5"
AUDIT_DESCRIPTION = "Read one HDF5 batch and report the implemented Retina SNN tensor flow."


class PipelineAuditError(ValueError):
    pass

def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    h5_path = args.h5.resolve()
    if not h5_path.is_file():
        raise PipelineAuditError(f"HDF5 file does not exist: {h5_path}")

    horizons = _parse_horizons(args.horizons)
    device = torch.device(args.device)
    torch.manual_seed(7)
    export = load_cone_response(h5_path)
    components = build_stage1_components(
        export.positions_degs,
        Stage1BuildConfig(
            dt_ms=dt_ms_from_time_axis_seconds(export.time_axis_seconds),
            horizon_count=len(horizons),
            eccentricity_deg=export.eccentricity_deg,
            midget_sampling=(
                MidgetSamplingMode.FOVEAL_PRIVATE_LINE
                if export.eccentricity_deg == 0
                else MidgetSamplingMode.CONVERGENT
            ),
        ),
    )
    mean, scale = fit_log_cone_stats((h5_path,))
    dataset = ISETBioH5Dataset(
        ISETBioH5DatasetConfig(
            h5_path=h5_path,
            input_steps=args.input_steps,
            horizons=horizons,
            target_fine_pool=components.target_pools.fine,
            target_coarse_pool=components.target_pools.coarse,
        ),
        ConeNormalizationStats(mean, scale),
    )
    batch = collate_isetbio_h5_batch([dataset[0]])
    core = components.core.to(device).eval()
    decoder = components.decoder.to(device).eval()

    with torch.inference_mode():
        history, state, diagnostics = core.forward_sequence(
            batch.x_cone.to(device),
            return_diagnostics=True,
        )
        prediction = decoder(_last_rgc_output(history))

    print("Retina Predictive SNN one-batch shape audit")
    print(f"h5={h5_path}")
    print(
        "metadata="
        f"T={export.response.shape[0]} Ncone={export.response.shape[1]} "
        f"dt_ms={dataset.dt_ms:g} eccentricity_deg={dataset.eccentricity_deg:g} "
        f"source_kind={export.stimulus_source_kind!r} source_id={export.source_id!r}"
    )
    print(f"windows={len(dataset)} horizons={dataset.horizons} device={device}")
    _print_tensor("raw cone_response_achromatic", torch.from_numpy(export.response))
    _print_tensor("cone_xy_deg", torch.from_numpy(export.positions_degs))
    _print_tensor("target fine pool", components.target_pools.fine)
    _print_tensor("target coarse pool", components.target_pools.coarse)
    _print_tensor("batch.x_cone", batch.x_cone)
    _print_tensor("batch.targets.fine", batch.targets.fine)
    _print_tensor("batch.targets.coarse", batch.targets.coarse)
    _print_tensor("H1 state (final)", state.h1)
    _print_tensor("BC output (final)", state.bipolar.output)
    _print_tensor("BC transient baseline (final)", state.bipolar.transient_baseline)
    _print_tensor("local AC state (final)", state.amacrine)
    _print_populations("RGC spike history", history.spikes)
    _print_populations("RGC rate history", history.rates)
    _print_populations("RGC membrane (final)", state.rgc.membrane)
    _print_populations("RGC adaptation (final)", state.rgc.adaptation)
    _print_populations("RGC rate state (final)", state.rgc.rate)
    _print_tensor("decoder prediction fine", prediction.target_fine)
    _print_tensor("decoder prediction coarse", prediction.target_coarse)
    _print_final_dynamics(diagnostics[-1])
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=AUDIT_DESCRIPTION)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--input-steps", type=int, default=8)
    parser.add_argument("--horizons", default="1,2,4")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.input_steps < 1:
        parser.error("--input-steps must be positive")
    return args


def _parse_horizons(raw: str) -> tuple[int, ...]:
    try:
        horizons = tuple(int(value) for value in raw.split(",") if value)
    except ValueError as exc:
        raise PipelineAuditError("--horizons must contain integers") from exc
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise PipelineAuditError("--horizons must contain positive offsets")
    return horizons


def _last_rgc_output(history: RGCOutput) -> RGCOutput:
    return RGCOutput(
        spikes=_last_populations(history.spikes),
        rates=_last_populations(history.rates),
    )


def _last_populations(populations: RGCPopulationTensors) -> RGCPopulationTensors:
    return RGCPopulationTensors(
        midget=populations.midget[:, -1],
        parasol=populations.parasol[:, -1],
        residual=populations.residual[:, -1],
    )


def _print_tensor(name: str, tensor: torch.Tensor) -> None:
    values = tensor.coalesce().values() if tensor.layout == torch.sparse_coo else tensor
    print(
        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} "
        f"min={values.min().item():.6g} max={values.max().item():.6g} "
        f"mean={values.mean().item():.6g} std={values.std(unbiased=False).item():.6g}"
    )


def _print_populations(prefix: str, populations: RGCPopulationTensors) -> None:
    _print_tensor(f"{prefix}.midget", populations.midget)
    _print_tensor(f"{prefix}.parasol", populations.parasol)
    _print_tensor(f"{prefix}.residual", populations.residual)


def _print_final_dynamics(diagnostics: RetinaStepDiagnostics) -> None:
    print("final-step dynamics (diagnostic scalars):")
    for group in ("h1", "bipolar", "amacrine", "rgc"):
        values = diagnostics[group]
        selected = {
            name: tensor
            for name, tensor in values.items()
            if name.endswith(("tau_ms", "leak", "g_ab", "g_ba", "g_ag", "kinetic_mix"))
        }
        for name, tensor in selected.items():
            print(f"  {name}={tensor.cpu().tolist()}")


if __name__ == "__main__":
    raise SystemExit(main())
