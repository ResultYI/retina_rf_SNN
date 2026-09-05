from __future__ import annotations

import ast
import json
import shutil
from dataclasses import fields
from pathlib import Path

import pytest

from evaluation.mechanistic_retina.metrics import JsonValue
from evaluation.model_comparison.sample_efficiency_reuse import (
    CANONICAL_REUSE_DIR,
    CanonicalReuseError,
    CanonicalReuseErrorCode,
    CanonicalReuseRequest,
    load_canonical_reuse_rows,
)


def test_load_canonical_reuse_rows_when_canonical_identity_matches() -> None:
    # Given: the unchanged canonical Candidate0 T=2 artifact directory.

    # When: metric-level reuse rows are loaded.
    rows = load_canonical_reuse_rows()

    # Then: exactly the canonical 100% run metrics are returned with provenance.
    counts = {model: sum(row.model == model for row in rows) for model in _models()}
    assert len(rows) == 33
    assert counts == {
        "Bias": 3,
        "GLM-SH": 3,
        "LN-LN": 9,
        "Graph-TCN": 9,
        "Mechanistic Retina": 9,
    }
    assert {row.fraction for row in rows} == {1.0}
    assert {row.count for row in rows} == {112}
    assert {row.reuse_status for row in rows} == {"metric_level_canonical_reuse"}
    assert {row.fairness_regime for row in rows} == {
        "architecture-size",
        "shared-reference",
    }
    assert all(row.source_run_id.startswith("canonical-model-comparison-t2:") for row in rows)
    assert all(row.provenance.artifact_sha256["per-run-metrics.csv"] for row in rows)

    mechanistic = next(
        row
        for row in rows
        if row.model == "Mechanistic Retina"
        and row.bank_seed == 31001
        and row.model_seed == 19
    )
    assert mechanistic.val_ce == pytest.approx(0.37048518657684326)
    assert mechanistic.global_rf == pytest.approx(0.877118547473536)
    assert mechanistic.exact_cell == pytest.approx(0.875)


@pytest.mark.parametrize(
    ("mutation", "artifact_name"),
    (
        ("candidate0_rf_sha256", "identity-manifest.json"),
        ("candidate0_source_sha256", "identity-manifest.json"),
        ("validation_cone_sha256", "identity-manifest.json"),
        ("model_seeds", "identity-manifest.json"),
        ("rf_estimand", "identity-manifest.json"),
        ("final_test_boundary", "identity-manifest.json"),
        ("per_run_row_count", "per-run-metrics.csv"),
    ),
)
def test_load_canonical_reuse_rows_when_artifact_identity_is_corrupt(
    tmp_path: Path, mutation: str, artifact_name: str
) -> None:
    # Given: a copied canonical directory with one identity-critical mutation.
    copied = tmp_path / "canonical-copy"
    shutil.copytree(CANONICAL_REUSE_DIR, copied)
    _mutate(copied, mutation)

    # When/Then: reuse fails closed before any training fallback can execute.
    with pytest.raises(CanonicalReuseError) as raised:
        load_canonical_reuse_rows(
            CanonicalReuseRequest(
                root=Path("."),
                canonical_dir=copied,
            )
        )
    assert raised.value.code is CanonicalReuseErrorCode.IDENTITY_MISMATCH
    assert artifact_name in str(raised.value.path)


def test_load_canonical_reuse_rows_when_validation_bank_hash_changes(
    tmp_path: Path,
) -> None:
    # Given: a copied manifest whose first bank validation hash no longer matches.
    copied = tmp_path / "canonical-copy"
    shutil.copytree(CANONICAL_REUSE_DIR, copied)
    manifest = _read_manifest(copied)
    banks = manifest["banks"]
    assert isinstance(banks, list)
    bank0 = banks[0]
    assert isinstance(bank0, dict)
    bank0["validation_sha256"] = "0" * 64
    _write_manifest(copied, manifest)

    # When/Then: the precise fail-closed code appears before the sentinel runs.
    with pytest.raises(CanonicalReuseError) as raised:
        load_canonical_reuse_rows(
            CanonicalReuseRequest(
                root=Path("."),
                canonical_dir=copied,
            )
        )
    assert raised.value.code is CanonicalReuseErrorCode.IDENTITY_MISMATCH
    assert str(raised.value).startswith("CANONICAL_REUSE_IDENTITY_MISMATCH")


def test_load_canonical_reuse_rows_when_called_outside_request_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a minimal copied repo root and an unrelated current working directory.
    copied_root = tmp_path / "copied-root"
    _copy_root_fixture(Path("."), copied_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)

    # When: reuse is loaded with root pointing at the copied fixture.
    rows = load_canonical_reuse_rows(
        CanonicalReuseRequest(root=copied_root, canonical_dir=CANONICAL_REUSE_DIR)
    )

    # Then: canonical config/source hashing honors the request root, not cwd.
    assert len(rows) == 33


def test_canonical_reuse_has_no_training_checkpoint_or_tensor_fallback() -> None:
    # Given: the public request type and production source for the reuse loader.
    source_path = Path("evaluation/model_comparison/sample_efficiency_reuse.py")
    module = ast.parse(source_path.read_text(encoding="utf-8"))

    # When: public fields, imports, and calls are inspected.
    request_fields = {field.name for field in fields(CanonicalReuseRequest)}
    imported_modules = {
        node.module
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_names = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    # Then: there is no fallback hook or dependency on training/checkpoint/tensor surfaces.
    assert request_fields == {"root", "canonical_dir"}
    assert not imported_modules & {
        "evaluation.model_comparison.training",
        "evaluation.model_comparison.baseline_runs",
        "evaluation.model_comparison.mechanistic_run",
        "evaluation.model_comparison.experiment",
        "torch",
    }
    assert "torch" not in imported_names
    assert not called_names & {
        "run_bias",
        "run_glm",
        "run_lnln",
        "run_graph_tcn",
        "run_mechanistic",
    }
    assert "checkpoint" not in source_path.read_text(encoding="utf-8").lower()


def _models() -> tuple[str, ...]:
    return ("Bias", "GLM-SH", "LN-LN", "Graph-TCN", "Mechanistic Retina")


def _copy_root_fixture(source_root: Path, target_root: Path) -> None:
    manifest = _read_manifest(source_root / CANONICAL_REUSE_DIR)
    shutil.copytree(source_root / CANONICAL_REUSE_DIR, target_root / CANONICAL_REUSE_DIR)
    config = json.loads((source_root / "configs/model_comparison_t2.yaml").read_text(encoding="utf-8"))
    _copy_relative(source_root, target_root, Path(str(config["candidate0_path"])))
    source_hashes = manifest["source_hashes"]
    assert isinstance(source_hashes, dict)
    for relative_path in source_hashes:
        _copy_relative(source_root, target_root, Path(relative_path))


def _copy_relative(source_root: Path, target_root: Path, relative_path: Path) -> None:
    target = target_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / relative_path, target)


def _read_manifest(canonical_dir: Path) -> dict[str, JsonValue]:  # noqa: DICT_OK
    return json.loads((canonical_dir / "identity-manifest.json").read_text(encoding="utf-8"))


def _write_manifest(canonical_dir: Path, manifest: dict[str, JsonValue]) -> None:  # noqa: DICT_OK
    (canonical_dir / "identity-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _mutate(canonical_dir: Path, mutation: str) -> None:
    if mutation == "per_run_row_count":
        path = canonical_dir / "per-run-metrics.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        return
    manifest = _read_manifest(canonical_dir)
    if mutation == "model_seeds":
        manifest["model_seeds"] = [19, 20, 22]
    elif mutation == "final_test_boundary":
        manifest["final_test_boundary"] = ["FINAL_TEST_WAS_USED"]
    elif mutation == "rf_estimand":
        manifest["rf_estimand"] = "static RF"
    elif mutation in {
        "candidate0_rf_sha256",
        "candidate0_source_sha256",
        "validation_cone_sha256",
    }:
        manifest[mutation] = "f" * 64
    else:
        raise AssertionError(mutation)
    _write_manifest(canonical_dir, manifest)
