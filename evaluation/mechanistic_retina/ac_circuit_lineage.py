from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def implementation_source_sha256() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    analysis_paths = (
        repo_root / "evaluation/mechanistic_retina/ac_circuit_inputs.py",
        repo_root / "evaluation/mechanistic_retina/ac_circuit_lineage.py",
        repo_root / "evaluation/mechanistic_retina/ac_circuit_perturbation.py",
        repo_root / "evaluation/mechanistic_retina/ac_circuit_support.py",
        repo_root / "evaluation/mechanistic_retina/rf_effective.py",
        repo_root / "evaluation/mechanistic_retina/spike_banks.py",
    )
    model_paths = tuple(sorted((repo_root / "models/mechanistic_retina").glob("*.py")))
    paths = tuple(sorted((*analysis_paths, *model_paths)))
    return {
        path.relative_to(repo_root).as_posix(): file_sha256(path) for path in paths
    }


__all__ = ["file_sha256", "implementation_source_sha256"]
