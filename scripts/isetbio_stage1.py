from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.isetbio_h5_gate import H5GateError, H5Summary, compare_reproducible, validate_hdf5

ROOT = Path(__file__).resolve().parents[1]
MATLAB_DIR = ROOT / "scripts" / "matlab"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class StageMinusOneError(RuntimeError):
    def __init__(self, layer: str, message: str) -> None:
        super().__init__(message)
        self.layer = layer


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    input_path: Path
    output_dir: Path
    manifest_csv: Path
    max_items: int
    time_steps: int
    dt_ms: float
    field_of_view_deg: float
    export_crop_fov_deg: float
    seed: int
    matlab_bin: str
    achromatic_stimulus_enabled: bool
    sequence_directory: bool
    sequence_root: bool
    reproducibility_check: bool


@dataclass(frozen=True, slots=True)
class ManifestRow:
    source_image_id: str
    source_image_path: Path
    output_h5_path: Path
    seed: int
    status: str
    failure_layer: str
    message: str


def load_generation_config(config_path: Path) -> GenerationConfig:
    values = _read_flat_yaml(config_path)
    base = config_path.resolve().parent
    input_path = _path_value(values, "input_path", base)
    output_dir = _path_value(values, "output_dir", base)
    field_of_view_deg = _float_value(values, "field_of_view_deg", 0.5)
    export_crop_fov_deg = _float_value(
        values,
        "export_crop_fov_deg",
        field_of_view_deg,
    )
    if export_crop_fov_deg > field_of_view_deg:
        raise StageMinusOneError(
            "config",
            "export_crop_fov_deg must not exceed field_of_view_deg",
        )
    sequence_directory = _bool_value(
        values,
        "treat_input_directory_as_sequence",
        False,
    )
    sequence_root = _bool_value(
        values,
        "treat_child_directories_as_sequences",
        False,
    )
    if sequence_directory and sequence_root:
        raise StageMinusOneError(
            "config",
            "Only one sequence input mode may be enabled",
        )
    return GenerationConfig(
        input_path=input_path,
        output_dir=output_dir,
        manifest_csv=_manifest_path(values, output_dir, base),
        max_items=_int_value(values, "max_items", 3),
        time_steps=_int_value(values, "time_steps", 16),
        dt_ms=_float_value(values, "dt_ms", 5.0),
        field_of_view_deg=field_of_view_deg,
        export_crop_fov_deg=export_crop_fov_deg,
        seed=_int_value(values, "seed", 7),
        matlab_bin=values.get("matlab_bin", "matlab"),
        achromatic_stimulus_enabled=_bool_value(values, "achromatic_stimulus_enabled", True),
        sequence_directory=sequence_directory,
        sequence_root=sequence_root,
        reproducibility_check=_bool_value(values, "reproducibility_check", True),
    )


def collect_sources(
    input_path: Path,
    *,
    max_items: int,
    sequence_directory: bool,
    sequence_root: bool = False,
) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if sequence_directory:
        if not input_path.is_dir():
            raise StageMinusOneError("input", f"sequence directory does not exist: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise StageMinusOneError("input", f"input_path does not exist: {input_path}")
    if sequence_root:
        sources = sorted(path for path in input_path.iterdir() if path.is_dir())
        if not sources:
            raise StageMinusOneError("input", f"no sequence directories found in: {input_path}")
        return sources[:max_items]
    sources = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not sources:
        raise StageMinusOneError("input", f"no image files found in: {input_path}")
    return sources[:max_items]


def matlab_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def run_matlab_check(config: GenerationConfig) -> None:
    expression = f"addpath({matlab_string(MATLAB_DIR)}); check_isetbio_env"
    _run_matlab(config.matlab_bin, expression)


def run_matlab_generation(
    config: GenerationConfig, config_path: Path, source: Path, output: Path, seed: int
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    expression = (
        f"addpath({matlab_string(MATLAB_DIR)}); "
        "generate_cone_h5_from_images("
        f"{matlab_string(source)}, {matlab_string(output)}, "
        f"{matlab_string(config_path.resolve())}, {seed})"
    )
    _run_matlab(config.matlab_bin, expression)


def generate_all(
    config_path: Path, config: GenerationConfig
) -> tuple[list[H5Summary], list[ManifestRow]]:
    sources = collect_sources(
        config.input_path,
        max_items=config.max_items,
        sequence_directory=config.sequence_directory,
        sequence_root=config.sequence_root,
    )
    summaries: list[H5Summary] = []
    rows: list[ManifestRow] = []
    for index, source in enumerate(sources):
        seed = config.seed + index
        output = config.output_dir / f"{source.stem}_seed{seed}.h5"
        try:
            run_matlab_generation(config, config_path, source, output, seed)
            summaries.append(validate_hdf5(output))
            rows.append(ManifestRow(source.name, source, output, seed, "ok", "", ""))
        except (StageMinusOneError, H5GateError) as exc:
            rows.append(ManifestRow(source.name, source, output, seed, "failed", exc.layer, str(exc)))
    if not summaries:
        write_manifest(config.manifest_csv, rows)
        raise StageMinusOneError("Stage -1", "no HDF5 passed readback")
    if config.reproducibility_check:
        repro = summaries[0].path.with_suffix(".repro.h5")
        run_matlab_generation(config, config_path, sources[0], repro, config.seed)
        validate_hdf5(repro)
        compare_reproducible(summaries[0].path, repro)
    return summaries, rows


def write_manifest(path: Path, rows: Sequence[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "source_image_id",
                "source_image_path",
                "output_h5_path",
                "seed",
                "status",
                "failure_layer",
                "message",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.source_image_id,
                    row.source_image_path,
                    row.output_h5_path,
                    row.seed,
                    row.status,
                    row.failure_layer,
                    row.message,
                )
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-env-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_generation_config(args.config)
        run_matlab_check(config)
        if args.check_env_only:
            return 0
        summaries, rows = generate_all(args.config, config)
        write_manifest(config.manifest_csv, rows)
    except (StageMinusOneError, H5GateError) as exc:
        print(f"Stage -1 failed at {exc.layer}: {exc}")
        return 1
    _print_success(summaries, config.manifest_csv)
    return 0


def _run_matlab(matlab_bin: str, expression: str) -> None:
    executable = shutil.which(matlab_bin) or matlab_bin
    try:
        result = subprocess.run(
            [executable, "-batch", expression],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="")
    except FileNotFoundError as exc:
        raise StageMinusOneError("MATLAB environment", f"MATLAB not found: {matlab_bin}") from exc
    except subprocess.CalledProcessError as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        raise StageMinusOneError(
            _classify_matlab_failure(output),
            f"MATLAB batch failed: {output.strip() or exc}",
        ) from exc


def _classify_matlab_failure(output: str) -> str:
    if "Fatal Startup Error" in output or "File system inconsistency" in output:
        return "MATLAB environment"
    if "ISETBioEnvironment" in output or "Missing MATLAB/ISETBio symbols" in output:
        return "ISETBio path"
    if "sceneFromFile" in output or "oiCompute" in output or "InvalidImage" in output:
        return "scene construction"
    if "cMosaic" in output or "ConeCountMismatch" in output or "UnsupportedConeTypes" in output:
        return "cMosaic"
    if "EyeMovement" in output or "fixEM" in output:
        return "eye movement"
    if "h5create" in output or "h5write" in output or "HDF5" in output:
        return "HDF5 export"
    return "MATLAB"


def _read_flat_yaml(path: Path) -> Mapping[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _path_value(values: Mapping[str, str], key: str, base: Path) -> Path:
    if key not in values:
        raise StageMinusOneError("config", f"missing required config key: {key}")
    path = Path(values[key])
    return path if path.is_absolute() else base / path


def _manifest_path(values: Mapping[str, str], output_dir: Path, base: Path) -> Path:
    if "manifest_csv" not in values:
        return output_dir / "manifest.csv"
    path = Path(values["manifest_csv"])
    return path if path.is_absolute() else base / path


def _int_value(values: Mapping[str, str], key: str, default: int) -> int:
    value = int(float(values.get(key, str(default))))
    if value <= 0:
        raise StageMinusOneError("config", f"{key} must be positive")
    return value


def _float_value(values: Mapping[str, str], key: str, default: float) -> float:
    value = float(values.get(key, str(default)))
    if not np.isfinite(value) or value <= 0:
        raise StageMinusOneError("config", f"{key} must be positive and finite")
    return value


def _bool_value(values: Mapping[str, str], key: str, default: bool) -> bool:
    return values.get(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _print_success(summaries: Sequence[H5Summary], manifest_csv: Path) -> None:
    for summary in summaries:
        print(
            "OK "
            f"{summary.path} "
            f"time={summary.time_steps} cones={summary.cone_count} "
            f"lms_shape={summary.lms_shape} "
            f"achromatic_shape={summary.achromatic_shape} "
            f"dt_ms={summary.dt_ms:.6g}"
        )
    print(f"manifest_csv={manifest_csv}")
    print("Stage -1 passed")
