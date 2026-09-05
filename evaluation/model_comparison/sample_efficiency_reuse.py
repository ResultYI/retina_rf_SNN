from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Final

from evaluation.mechanistic_retina.metrics import JsonValue
from evaluation.model_comparison.sample_efficiency_reporting import Profile

CANONICAL_REUSE_DIR: Final = Path(".omo/evidence/canonical-model-comparison-t2")

_BOUNDARY: Final = (
    "TEST_SPLIT_ACCESSED_FOR_IDENTITY_ONLY",
    "TEST_EXAMPLES_NOT_USED_FOR_INFERENCE_METRICS_MODEL_SELECTION_OR_CONCLUSIONS",
    "FINAL_TEST_SCIENTIFIC_EVALUATION_NOT_CONSUMED",
)
_ARTIFACT_SHA256: Final = {
    "identity-manifest.json": "13165c2aa4c33863883c1cee77ece01bfe6fd2e3f153dd8427f89631579414c7",
    "per-run-metrics.csv": "06529f0aba04ec06362b41ad2a2c63f7ed4b746bbc2ca77cd1cd4012d57756e8",
    "per-cell-metrics.csv": "25a9f32cade25e903326aa4369bc1df926426fb04f0d4d86acaf62938f6c61ce",
    "model-comparison.csv": "eee098205fe46a4937b590802a69e32c35448404c188639fd9e79c4340e802ac",
    "stability-results.json": "977e5c1e759625b4946f80e6744261d4a24657cb5750f2c1b5dea2fe73eb1302",
    "experiment-config.yaml": "d4baab5a36f77665bf3107cf069ec117bab6aaa55fea2b3167baff3cb3785c30",
}
_MODEL_COUNTS: Final = {"Bias": 3, "GLM-SH": 3, "LN-LN": 9, "Graph-TCN": 9, "Mechanistic Retina": 9}
_MODEL_CELL_ROWS: Final = {model: count * 16 for model, count in _MODEL_COUNTS.items()}
_BANK_HASHES: Final = {
    31001: ("2f213dff34c9c6d16eee94cadfc08cc7754366eb10e2e8d86d997249a809d664", "7b70b62162df7444f0d65ed93e603dc39ed9cd24c8597273c6e7142b9f54dcf1"),
    31002: ("856aafebdddccf9d01ab6efc745145ac4e42d0eed34315ebebfcb902c4178e4b", "6cc1d18d4ea85d9a6f15a88a6cc3780db7e65d820a167a47a5a808d3ac40a11a"),
    31003: ("2ce2d0c88e887f93696969c749508821fa7f4325f523cb75c4303dbaa0d17a9f", "6e37f12348798c3342a9f20b23c4443ada81c64d6039d7cd0e505f0e52f0e161"),
}
_RUN_FIELDS: Final = "model,bank_seed,model_seed,params,val_ce,sampled_nll,bits_per_spike,logit_rmse,brier,global_rf,spatial,temporal,exact_cell,nearest_type_polarity,prototype_centroid,gradients_finite".split(",")


@unique
class CanonicalReuseErrorCode(StrEnum):
    IDENTITY_MISMATCH = "CANONICAL_REUSE_IDENTITY_MISMATCH"


@dataclass(frozen=True, slots=True)
class CanonicalReuseError(Exception):
    code: CanonicalReuseErrorCode
    path: Path
    detail: str

    def __str__(self) -> str:
        return f"{self.code.value}: {self.detail}: {self.path}"


@dataclass(frozen=True, slots=True)
class CanonicalReuseRequest:
    root: Path = Path(".")
    canonical_dir: Path = CANONICAL_REUSE_DIR


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    artifact_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class CanonicalReuseRow:
    model: str
    bank_seed: int
    model_seed: int | None
    parameter_count: int
    val_ce: float
    sampled_nll: float
    bits_per_spike: float
    logit_rmse: float
    brier: float
    global_rf: float | None
    spatial: float | None
    temporal: float | None
    exact_cell: float | None
    nearest_type_polarity: float | None
    prototype_centroid: float | None
    gradients_finite: bool
    fraction: float
    count: int
    fairness_regime: Profile
    reuse_status: str
    source_run_id: str
    provenance: ArtifactProvenance


@dataclass(frozen=True, slots=True)
class _LoadedArtifacts:
    root: Path
    canonical_dir: Path
    manifest: JsonValue
    config: JsonValue
    per_run: tuple[Mapping[str, str], ...]
    per_cell: tuple[Mapping[str, str], ...]
    summary: tuple[Mapping[str, str], ...]
    stability: JsonValue


def load_canonical_reuse_rows(
    request: CanonicalReuseRequest | None = None,
) -> tuple[CanonicalReuseRow, ...]:
    active = CanonicalReuseRequest() if request is None else request
    canonical_dir = _resolve(active.root, active.canonical_dir)
    loaded = _LoadedArtifacts(
        active.root,
        canonical_dir,
        _read_json(canonical_dir / "identity-manifest.json"),
        _read_json(canonical_dir / "experiment-config.yaml"),
        _read_csv(canonical_dir / "per-run-metrics.csv", _RUN_FIELDS),
        _read_csv(canonical_dir / "per-cell-metrics.csv", ()),
        _read_csv(canonical_dir / "model-comparison.csv", ()),
        _read_json(canonical_dir / "stability-results.json"),
    )
    _validate_manifest(active.root, loaded)
    _validate_counts(loaded)
    provenance = _validate_artifact_sha256(canonical_dir)
    return tuple(_reuse_row(row, provenance) for row in loaded.per_run)


def _resolve(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def _read_json(path: Path) -> JsonValue:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise _mismatch(path, "required artifact missing") from exc
    except json.JSONDecodeError as exc:
        raise _mismatch(path, "malformed JSON artifact") from exc


def _read_csv(path: Path, fields: Sequence[str]) -> tuple[Mapping[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if fields and tuple(reader.fieldnames or ()) != tuple(fields):
                raise _mismatch(path, "unexpected CSV schema")
            return tuple({key: value or "" for key, value in row.items()} for row in reader)
    except FileNotFoundError as exc:
        raise _mismatch(path, "required artifact missing") from exc


def _validate_manifest(root: Path, loaded: _LoadedArtifacts) -> None:
    manifest_path = loaded.canonical_dir / "identity-manifest.json"
    config_path = loaded.canonical_dir / "experiment-config.yaml"
    manifest = _mapping(manifest_path, loaded.manifest)
    config_map = _mapping(config_path, loaded.config)
    _expect(manifest, "lineage", "candidate0-t2-mechanism-identifiable-canonical-v1", manifest_path)
    _expect(manifest, "candidate0_rf_sha256", _item(config_map, "candidate0_rf_sha256", config_path), manifest_path)
    _expect(manifest, "candidate0_rf_sha256", "d21a877e80b70755325a0d909f6033c1dddb4f35dfb0bb28cbf16c16ad422921", manifest_path)
    _expect(manifest, "candidate0_source_sha256", _sha256(root / str(_item(config_map, "candidate0_path", config_path))), manifest_path)
    _expect(manifest, "train_cone_sha256", "ef7dddc7db4cd76f417a8c4b919e5e594e855605ce1b9338d45ddf431e0c6d44", manifest_path)
    _expect(manifest, "validation_cone_sha256", "fd2d660c47c4ee180f154b5490ed6a1be4a51766a6b8ca42bbcd971c93cb53a1", manifest_path)
    _expect(manifest, "trial_budget", 2, manifest_path)
    _expect(manifest, "model_seeds", [19, 20, 21], manifest_path)
    _expect(manifest, "rf_estimand", "conditional total-dynamic logit RF, 16 lags", manifest_path)
    _expect(manifest, "final_test_boundary", list(_BOUNDARY), manifest_path)
    _validate_config(loaded, config_map, manifest)
    _validate_banks(manifest_path, _sequence(manifest, "banks", manifest_path))
    _validate_source_hashes(root, manifest_path, _mapping_key(manifest, "source_hashes", manifest_path))


def _validate_config(
    loaded: _LoadedArtifacts,
    config: Mapping[str, JsonValue],
    manifest: Mapping[str, JsonValue],
) -> None:
    config_path = loaded.canonical_dir / "experiment-config.yaml"
    _expect(config, "data_seed", 19, config_path)
    _expect(config, "bank_seeds", [31001, 31002, 31003], config_path)
    _expect(config, "model_seeds", [19, 20, 21], config_path)
    _expect(config, "trials", 2, config_path)
    _expect(config, "steps", 400, config_path)
    source_hashes = _mapping_key(manifest, "source_hashes", loaded.canonical_dir / "identity-manifest.json")
    _expect(source_hashes, "configs\\model_comparison_t2.yaml", _sha256(loaded.root / "configs/model_comparison_t2.yaml"), config_path)


def _validate_banks(manifest_path: Path, banks: Sequence[JsonValue]) -> None:
    if len(banks) != len(_BANK_HASHES):
        raise _mismatch(manifest_path, "unexpected bank count")
    for bank in banks:
        bank_map = _mapping(manifest_path, bank)
        seed = int(bank_map["seed"])
        if seed not in _BANK_HASHES:
            raise _mismatch(manifest_path, "unexpected bank seed")
        expected = _BANK_HASHES[seed]
        if (bank_map["train_sha256"], bank_map["validation_sha256"]) != expected:
            raise _mismatch(manifest_path, "bank hash mismatch")
        _expect(bank_map, "trials", 2, manifest_path)


def _validate_source_hashes(root: Path, manifest_path: Path, source_hashes: Mapping[str, JsonValue]) -> None:
    for rel_path, expected in source_hashes.items():
        path = root / rel_path
        if expected != _sha256(path):
            raise _mismatch(manifest_path, f"source hash mismatch for {rel_path}")


def _validate_counts(loaded: _LoadedArtifacts) -> None:
    counts = {model: sum(row["model"] == model for row in loaded.per_run) for model in _MODEL_COUNTS}
    if counts != _MODEL_COUNTS or len(loaded.per_run) != 33:
        raise _mismatch(loaded.canonical_dir / "per-run-metrics.csv", f"run counts {counts}")
    cell_counts = {model: sum(row["model"] == model for row in loaded.per_cell) for model in _MODEL_COUNTS}
    if cell_counts != _MODEL_CELL_ROWS or len(loaded.per_cell) != 528:
        raise _mismatch(loaded.canonical_dir / "per-cell-metrics.csv", f"cell counts {cell_counts}")
    summary_counts = {row["model"]: int(row["runs"]) for row in loaded.summary}
    if summary_counts != _MODEL_COUNTS or len(loaded.summary) != 5:
        raise _mismatch(loaded.canonical_dir / "model-comparison.csv", f"summary counts {summary_counts}")
    stability_map = _mapping(loaded.canonical_dir / "stability-results.json", loaded.stability)
    for model, count in _MODEL_COUNTS.items():
        model_stability = _mapping_key(stability_map, model, loaded.canonical_dir / "stability-results.json")
        _expect(model_stability, "run_count", count, loaded.canonical_dir / "stability-results.json")


def _validate_artifact_sha256(canonical_dir: Path) -> ArtifactProvenance:
    hashes = {name: _sha256(canonical_dir / name) for name in _ARTIFACT_SHA256}
    for name, expected in _ARTIFACT_SHA256.items():
        if hashes[name] != expected:
            raise _mismatch(canonical_dir / name, "artifact provenance sha256 mismatch")
    return ArtifactProvenance(hashes)


def _reuse_row(row: Mapping[str, str], provenance: ArtifactProvenance) -> CanonicalReuseRow:
    model = row["model"]
    seed = _optional_int(row["model_seed"])
    return CanonicalReuseRow(
        model,
        int(row["bank_seed"]),
        seed,
        int(row["params"]),
        float(row["val_ce"]),
        float(row["sampled_nll"]),
        float(row["bits_per_spike"]),
        float(row["logit_rmse"]),
        float(row["brier"]),
        _optional_float(row["global_rf"]),
        _optional_float(row["spatial"]),
        _optional_float(row["temporal"]),
        _optional_float(row["exact_cell"]),
        _optional_float(row["nearest_type_polarity"]),
        _optional_float(row["prototype_centroid"]),
        row["gradients_finite"] == "True",
        1.0,
        112,
        Profile.SHARED_REFERENCE if model in {"Bias", "GLM-SH"} else Profile.ARCHITECTURE_SIZE,
        "metric_level_canonical_reuse",
        f"canonical-model-comparison-t2:{model}:bank-{row['bank_seed']}:seed-{seed or 'none'}",
        provenance,
    )


def _optional_int(value: str) -> int | None:
    return None if value == "" else int(value)


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _mapping(path: Path, value: JsonValue) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict):
        raise _mismatch(path, "expected JSON object")
    return value


def _mapping_key(value: Mapping[str, JsonValue], key: str, path: Path) -> Mapping[str, JsonValue]:
    return _mapping(path, _item(value, key, path))


def _sequence(value: Mapping[str, JsonValue], key: str, path: Path) -> Sequence[JsonValue]:
    item = _item(value, key, path)
    if not isinstance(item, list):
        raise _mismatch(path, f"expected list at {key}")
    return item


def _expect(value: Mapping[str, JsonValue], key: str, expected: JsonValue, path: Path) -> None:
    if _item(value, key, path) != expected:
        raise _mismatch(path, f"{key} mismatch")


def _item(value: Mapping[str, JsonValue], key: str, path: Path) -> JsonValue:
    try:
        return value[key]
    except KeyError as exc:
        raise _mismatch(path, f"missing {key}") from exc


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError as exc:
        raise _mismatch(path, "required source/artifact missing") from exc


def _mismatch(path: Path, detail: str) -> CanonicalReuseError:
    return CanonicalReuseError(CanonicalReuseErrorCode.IDENTITY_MISMATCH, path, detail)


__all__ = ["ArtifactProvenance", "CANONICAL_REUSE_DIR", "CanonicalReuseError", "CanonicalReuseErrorCode", "CanonicalReuseRequest", "CanonicalReuseRow", "load_canonical_reuse_rows"]
