from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from evaluation.mechanistic_retina.rf_base import (
    Candidate0LoadError,
    CandidateTeacherUsage,
    load_candidate0,
)


def test_failed_preflight_candidate_requires_explicit_reference_usage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "failed-preflight.json"
    _write_candidate_artifact(
        path,
        selected_candidate=None,
        case="TEACHER-PREFLIGHT-FAILED",
    )

    with pytest.raises(Candidate0LoadError, match="no selected candidate"):
        load_candidate0(path)

    with pytest.raises(Candidate0LoadError, match="reference candidate index"):
        load_candidate0(
            path,
            usage=CandidateTeacherUsage.DEVELOPMENT_REFERENCE,
        )

    reference = load_candidate0(
        path,
        usage=CandidateTeacherUsage.DEVELOPMENT_REFERENCE,
        reference_candidate_index=0,
    )
    assert reference.teacher_usage is CandidateTeacherUsage.DEVELOPMENT_REFERENCE
    assert not reference.preflight_passed
    assert reference.selected_candidate is None
    assert reference.loaded_candidate_index == 0
    assert reference.artifact_case == "TEACHER-PREFLIGHT-FAILED"

    selected_path = tmp_path / "failed-preflight-with-selection.json"
    _write_candidate_artifact(
        selected_path,
        selected_candidate=0,
        case="TEACHER-PREFLIGHT-FAILED",
    )
    with pytest.raises(Candidate0LoadError, match="inconsistent selection"):
        load_candidate0(selected_path)

    failed_row_path = tmp_path / "failed-row-preflight.json"
    _write_candidate_artifact(
        failed_row_path,
        selected_candidate=0,
        case="TEACHER-PREFLIGHT-COMPLETE",
        candidate_passed=False,
    )
    with pytest.raises(Candidate0LoadError, match="invalid selection"):
        load_candidate0(failed_row_path)

    unknown_case_path = tmp_path / "unknown-preflight-case.json"
    _write_candidate_artifact(
        unknown_case_path,
        selected_candidate=0,
        case="TEACHER-PREFLIGHT-UNKNOWN",
        candidate_passed=True,
    )
    with pytest.raises(Candidate0LoadError, match="unsupported preflight case"):
        load_candidate0(unknown_case_path)

    invalid_index_path = tmp_path / "invalid-candidate-index.json"
    _write_candidate_artifact(
        invalid_index_path,
        selected_candidate=0,
        case="TEACHER-PREFLIGHT-COMPLETE",
        candidate_passed=True,
        candidate_index=False,
    )
    with pytest.raises(Candidate0LoadError, match="row index is invalid"):
        load_candidate0(invalid_index_path)

    missing_result_path = tmp_path / "missing-result.json"
    _write_candidate_artifact(
        missing_result_path,
        selected_candidate=None,
        case="TEACHER-PREFLIGHT-FAILED",
        include_result=False,
    )
    with pytest.raises(Candidate0LoadError, match="result status is invalid"):
        load_candidate0(
            missing_result_path,
            usage=CandidateTeacherUsage.DEVELOPMENT_REFERENCE,
            reference_candidate_index=0,
        )

    non_boolean_result_path = tmp_path / "non-boolean-result.json"
    _write_candidate_artifact(
        non_boolean_result_path,
        selected_candidate=None,
        case="TEACHER-PREFLIGHT-FAILED",
        candidate_passed=1,
    )
    with pytest.raises(Candidate0LoadError, match="result status is invalid"):
        load_candidate0(
            non_boolean_result_path,
            usage=CandidateTeacherUsage.DEVELOPMENT_REFERENCE,
            reference_candidate_index=0,
        )


def _write_candidate_artifact(
    path: Path,
    *,
    selected_candidate: int | None,
    case: str,
    candidate_passed: bool | int | None = None,
    candidate_index: int = 0,
    include_result: bool = True,
) -> None:
    rf = torch.arange(16 * 16 * 29, dtype=torch.float32).reshape(16, 16, 29)
    digest = hashlib.sha256(rf.numpy().tobytes()).hexdigest()
    metadata = [
        {
            "cell_id": f"cell-{index}",
            "type_id": "midget",
            "polarity": "ON",
            "position_x": float(index),
            "position_y": 0.0,
            "replicate_id": "r0",
        }
        for index in range(16)
    ]
    candidate = {
        "config": {"candidate_index": candidate_index},
        "rf_tensor": rf[None, None].tolist(),
        "rf_sha256": digest,
        "metadata": metadata,
    }
    if include_result:
        candidate["result"] = {
            "passed": (
                not case.endswith("FAILED")
                if candidate_passed is None
                else candidate_passed
            )
        }
    payload = {
        "schema": "static-teacher-preflight-v1",
        "case": case,
        "complete": True,
        "selected_candidate": selected_candidate,
        "candidates": [candidate],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
