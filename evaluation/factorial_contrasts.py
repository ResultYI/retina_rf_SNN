from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class FactorialContrastError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FactorialContrasts:
    common: float
    type: float
    polarity: float
    interaction: float


def factorial_contrasts(gains: np.ndarray) -> FactorialContrasts:
    values = np.asarray(gains, dtype=np.float64).reshape(-1)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise FactorialContrastError("Factorial gains must contain four finite cells")
    on_midget, off_midget, on_parasol, off_parasol = values
    return FactorialContrasts(
        common=float(values.mean()),
        type=float((-on_midget - off_midget + on_parasol + off_parasol) / 4),
        polarity=float((-on_midget + off_midget - on_parasol + off_parasol) / 4),
        interaction=float((on_midget - off_midget - on_parasol + off_parasol) / 4),
    )


__all__ = [
    "FactorialContrastError",
    "FactorialContrasts",
    "factorial_contrasts",
]
