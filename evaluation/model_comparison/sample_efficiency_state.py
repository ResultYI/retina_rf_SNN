from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from evaluation.mechanistic_retina.metrics import JsonValue
from evaluation.model_comparison.artifacts import validate_artifact_set, write_json

_CACHE_EXECUTION_SOURCES: Final = {
    "evaluation\\model_comparison\\sample_efficiency_runner.py",
    "evaluation\\model_comparison\\sample_efficiency_state.py",
    "scripts\\run_sample_efficiency.py",
}


@dataclass(frozen=True, slots=True)
class SampleEfficiencyStateError(RuntimeError):
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass(frozen=True, slots=True)
class CacheRequest:
    path: Path
    identity: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class FinalArtifactRequest:
    output: Path
    required: Sequence[str]
    identity: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class FinalArtifactState:
    output_dir: Path
    artifact_sha256: Mapping[str, str]


class ExclusiveLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> None:
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise SampleEfficiencyStateError("RUN_LOCKED", str(self.path)) from exc
        os.write(self.fd, str(os.getpid()).encode("utf-8"))

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.fd is None:
            raise SampleEfficiencyStateError("RUN_LOCK_RELEASE_FAILED", str(self.path))
        os.close(self.fd)
        self.path.unlink(missing_ok=True)


def cache_path(root: Path, run_dir: Path, fraction: float) -> Path:
    run_root = run_dir if run_dir.is_absolute() else root / run_dir
    return run_root / f"fraction-{int(fraction * 100):03d}" / "metrics-cache.json"


def read_cache(request: CacheRequest) -> tuple[Mapping[str, JsonValue], ...] | None:
    if not request.path.exists():
        return None
    try:
        payload = json.loads(request.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SampleEfficiencyStateError("STALE_FRACTION_CACHE", str(request.path)) from exc
    if not isinstance(payload, dict):
        raise SampleEfficiencyStateError("STALE_FRACTION_CACHE", str(request.path))
    rows = payload.get("rows")
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise SampleEfficiencyStateError("STALE_FRACTION_CACHE", str(request.path))
    if not _cache_identity_matches(identity, request.identity) or payload.get("complete") is not True or not isinstance(rows, list):
        raise SampleEfficiencyStateError("STALE_FRACTION_CACHE", str(request.path))
    if not all(isinstance(row, dict) for row in rows):
        raise SampleEfficiencyStateError("STALE_FRACTION_CACHE", str(request.path))
    return tuple(dict(row) for row in rows)


def write_cache(request: CacheRequest, rows: Sequence[Mapping[str, JsonValue]]) -> None:
    request.path.parent.mkdir(parents=True, exist_ok=True)
    temporary = request.path.with_suffix(".tmp")
    write_json(temporary, {"complete": True, "identity": dict(request.identity), "rows": rows})
    temporary.replace(request.path)


def completed_final(request: FinalArtifactRequest) -> FinalArtifactState | None:
    if not request.output.exists():
        return None
    validate_final_output(request)
    return FinalArtifactState(request.output, artifact_hashes(request.output, request.required))


def prepare_staging(output: Path) -> Path:
    staging = output.parent / f"{output.name}.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def promote_staging(staging: Path, request: FinalArtifactRequest, replace_existing: bool = False) -> None:
    try:
        validate_final_output(FinalArtifactRequest(staging, request.required, request.identity))
    except (OSError, ValueError) as exc:
        shutil.rmtree(staging)
        raise SampleEfficiencyStateError("STALE_FINAL_OUTPUT", str(staging)) from exc
    if request.output.exists():
        if not replace_existing:
            raise SampleEfficiencyStateError("STALE_FINAL_OUTPUT", str(request.output))
        shutil.rmtree(request.output)
    staging.replace(request.output)


def validate_final_output(request: FinalArtifactRequest) -> None:
    try:
        actual = {path.name for path in request.output.iterdir() if path.is_file()}
        if actual != set(request.required):
            raise SampleEfficiencyStateError("STALE_FINAL_OUTPUT", str(request.output))
        validate_artifact_set(request.output, request.required)
        identity = json.loads((request.output / "identity-manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SampleEfficiencyStateError("STALE_FINAL_OUTPUT", str(request.output)) from exc
    if identity != request.identity:
        raise SampleEfficiencyStateError("STALE_FINAL_OUTPUT", str(request.output))


def artifact_hashes(output: Path, required: Sequence[str]) -> Mapping[str, str]:
    return {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in required}


def _cache_identity_matches(actual: Mapping[str, JsonValue], expected: Mapping[str, JsonValue]) -> bool:
    return actual == expected or _cache_scientific_identity(actual) == _cache_scientific_identity(expected)


def _cache_scientific_identity(identity: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    source = identity.get("source_sha256")
    if not isinstance(source, dict):
        return identity
    kept = {key: value for key, value in source.items() if key not in _CACHE_EXECUTION_SOURCES}
    return {**identity, "source_sha256": kept}


__all__ = [
    "CacheRequest",
    "ExclusiveLock",
    "FinalArtifactRequest",
    "FinalArtifactState",
    "SampleEfficiencyStateError",
    "artifact_hashes",
    "cache_path",
    "completed_final",
    "prepare_staging",
    "promote_staging",
    "read_cache",
    "validate_final_output",
    "write_cache",
]
