from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from evaluation.rf_dynamic_teacher import (
    RecoveryContract,
    TeacherDynamicReference,
    align_teacher_dynamic_rf,
    classify_teacher_status,
    teacher_recovery_errors,
)


def test_signed_teacher_alignment_rejects_sign_swapped_gain() -> None:
    teacher = TeacherDynamicReference(
        low_kernel=torch.ones(2, 2, 1),
        high_kernel=torch.tensor([[[0.7], [0.7]], [[1.3], [1.3]]]),
    )

    alignment = align_teacher_dynamic_rf(
        torch.ones(2, 2, 1),
        torch.tensor([[[1.3], [1.3]], [[0.7], [0.7]]]),
        teacher,
    )

    assert alignment.status == "teacher_mismatch"
    assert alignment.direction_agreement == (False, False)


def test_signed_teacher_alignment_rejects_excessive_shape_for_pure_gain() -> None:
    teacher = TeacherDynamicReference(
        low_kernel=torch.ones(1, 2, 2),
        high_kernel=torch.full((1, 2, 2), 1.3),
    )

    alignment = align_teacher_dynamic_rf(
        torch.ones(1, 2, 2),
        torch.tensor([[[1.3, -1.3], [1.3, -1.3]]]),
        teacher,
    )

    assert alignment.status == "teacher_mismatch"
    assert alignment.excessive_shape_deformation


def test_signed_teacher_alignment_marks_degenerate_teacher_not_identifiable() -> None:
    teacher = TeacherDynamicReference(
        low_kernel=torch.ones(1, 2, 1),
        high_kernel=torch.ones(1, 2, 1),
    )

    alignment = align_teacher_dynamic_rf(
        torch.ones(1, 2, 1),
        torch.ones(1, 2, 1),
        teacher,
    )

    assert alignment.status == "not_identifiable"


def test_teacher_reference_flips_persisted_causal_lags_for_alignment() -> None:
    causal_high = torch.tensor([[[1.0], [2.0], [4.0]]])
    predicted_high = torch.flip(causal_high, dims=(1,))
    predicted_low = torch.zeros_like(predicted_high)
    unflipped_cosine = F.cosine_similarity(
        (predicted_high - predicted_low).flatten(1),
        causal_high.flatten(1),
        dim=1,
    )

    alignment = align_teacher_dynamic_rf(
        predicted_low,
        predicted_high,
        TeacherDynamicReference(
            low_kernel=torch.zeros_like(causal_high),
            high_kernel=causal_high,
        ),
    )

    assert float(1 - unflipped_cosine[0]) > 0.1
    assert alignment.kernel_delta_cosine_distance < 1e-6
    assert alignment.status == "supported"


def test_teacher_status_preserves_failed_state_reset_gate() -> None:
    alignment = align_teacher_dynamic_rf(
        torch.ones(1, 2, 1),
        torch.full((1, 2, 1), 1.2),
        TeacherDynamicReference(
            low_kernel=torch.ones(1, 2, 1),
            high_kernel=torch.full((1, 2, 1), 1.2),
        ),
    )

    assert classify_teacher_status("not_supported", [alignment]) == "not_supported"
    assert classify_teacher_status("not_identifiable", [alignment]) == "not_identifiable"


def test_teacher_recovery_error_uses_context_gain_envelope() -> None:
    envelope = torch.ones(2, 5, 1)
    envelope[1, :, 0] = torch.tensor([1.8, 1.4, 1.2, 1.1, 1.05])
    reference = TeacherDynamicReference(
        low_kernel=torch.ones(1, 2, 1),
        high_kernel=torch.full((1, 2, 1), 1.2),
        context_gain_envelope=envelope,
    )

    errors = teacher_recovery_errors(
        (((math.log(1.05),), (math.log(1.025),)),),
        reference,
        RecoveryContract(delays_ms=(0, 5), dt_ms=5.0),
    )

    assert errors[0] < 1e-6


def test_teacher_recovery_error_samples_are_source_level() -> None:
    envelope = torch.ones(10, 5, 1)
    envelope[1::2, :, 0] = torch.tensor([1.8, 1.4, 1.2, 1.1, 1.05])
    reference = TeacherDynamicReference(
        low_kernel=torch.ones(1, 2, 1),
        high_kernel=torch.full((1, 2, 1), 1.2),
        context_gain_envelope=envelope,
    )
    expected = ((math.log(1.05),), (math.log(1.025),))
    changed = [expected for _ in range(5)]
    changed[2] = ((math.log(1.15),), expected[1])

    errors = teacher_recovery_errors(
        tuple(changed),
        reference,
        RecoveryContract(delays_ms=(0, 5), dt_ms=5.0),
    )
    more_delay_errors = teacher_recovery_errors(
        tuple(
            (
                (math.log(1.05),),
                (math.log(1.025),),
                (math.log(1.0125),),
            )
            for _ in range(5)
        ),
        reference,
        RecoveryContract(delays_ms=(0, 5, 10), dt_ms=5.0),
    )

    assert len(errors) == 5
    assert errors[0] < 1e-6
    assert errors[2] > 0.0
    assert len(more_delay_errors) == 5
