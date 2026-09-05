from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


try:
    from numba import njit
except ImportError:
    def njit(function: Callable[..., object]) -> Callable[..., object]:
        return function


@dataclass(frozen=True, slots=True)
class Ran1State:
    state: int
    iy: int
    table: np.ndarray


def initialize_ran1(seed: int) -> Ran1State:
    ia = 16807
    im = 2147483647
    iq = 127773
    ir = 2836
    ntab = 32
    state = max(-seed, 1)
    table = np.zeros(ntab, dtype=np.int64)
    for warmup in range(ntab + 7, -1, -1):
        quotient = state // iq
        state = ia * (state - quotient * iq) - ir * quotient
        if state < 0:
            state += im
        if warmup < ntab:
            table[warmup] = state
    return Ran1State(state, int(table[0]), table)


@njit
def _advance_ran1(
    state: int, iy: int, table: np.ndarray, count: int
) -> tuple[np.ndarray, int, int, np.ndarray]:
    ia = 16807
    im = 2147483647
    iq = 127773
    ir = 2836
    ndiv = 1 + (im - 1) // 32
    output = np.empty(count, dtype=np.int8)
    for index in range(count):
        quotient = state // iq
        state = ia * (state - quotient * iq) - ir * quotient
        if state < 0:
            state += im
        table_index = iy // ndiv
        iy = int(table[table_index])
        table[table_index] = state
        output[index] = 1 if iy / im >= 0.5 else -1
    return output, state, iy, table


def recreate_binary_white_noise_block(
    ran1_state: Ran1State,
    width: int,
    height: int,
    frame_count: int,
) -> tuple[np.ndarray, Ran1State]:
    if width < 1 or height < 1 or frame_count < 1:
        raise ValueError("stimulus dimensions must be positive")
    values, state, iy, table = _advance_ran1(
        ran1_state.state,
        ran1_state.iy,
        ran1_state.table.copy(),
        width * height * frame_count,
    )
    frame_x_y = values.reshape(frame_count, width, height)
    stimulus = np.transpose(frame_x_y, (0, 2, 1))
    return stimulus, Ran1State(state, iy, table)


def recreate_binary_white_noise(
    seed: int,
    width: int,
    height: int,
    frame_count: int,
) -> np.ndarray:
    stimulus, _ = recreate_binary_white_noise_block(
        initialize_ran1(seed), width, height, frame_count
    )
    return stimulus


__all__ = [
    "Ran1State",
    "initialize_ran1",
    "recreate_binary_white_noise",
    "recreate_binary_white_noise_block",
]
