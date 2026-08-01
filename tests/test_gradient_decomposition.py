from __future__ import annotations

import pytest
import torch

from evaluation import parameter_audit


def test_type_differential_reports_opposed_type_gradients() -> None:
    # Given
    midget = torch.tensor([1.0, 0.0])
    parasol = torch.tensor([-1.0, 0.0])

    # When
    result = parameter_audit.type_differential(midget, parasol)

    # Then
    assert result.separation_ratio == pytest.approx(1.0)
    assert result.opposition_cosine == pytest.approx(1.0)
