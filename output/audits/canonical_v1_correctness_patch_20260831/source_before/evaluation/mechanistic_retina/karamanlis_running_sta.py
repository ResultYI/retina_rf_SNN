from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class StreamingSTA:
    numerator: np.ndarray
    spike_denominator: np.ndarray
    stimulus_sum: np.ndarray
    frame_denominator: np.ndarray
    height: int
    width: int

    @classmethod
    def empty(
        cls,
        lag_count: int,
        cell_count: int,
        height: int,
        width: int,
    ) -> StreamingSTA:
        if min(lag_count, cell_count, height, width) < 1:
            raise ValueError("STA dimensions must be positive")
        pixel_count = height * width
        return cls(
            numerator=np.zeros((cell_count, lag_count, pixel_count), dtype=np.float32),
            spike_denominator=np.zeros((cell_count, lag_count), dtype=np.float64),
            stimulus_sum=np.zeros((lag_count, pixel_count), dtype=np.float64),
            frame_denominator=np.zeros(lag_count, dtype=np.int64),
            height=height,
            width=width,
        )

    def add(self, stimulus: np.ndarray, response: np.ndarray) -> None:
        if stimulus.ndim != 3 or response.ndim != 2:
            raise ValueError("stimulus must be T×Y×X and response T×cells")
        if stimulus.shape[0] != response.shape[0]:
            raise ValueError("stimulus and response must share time")
        if stimulus.shape[1:] != (self.height, self.width):
            raise ValueError("stimulus spatial shape changed between trials")
        if response.shape[1] != self.numerator.shape[0]:
            raise ValueError("response cell count changed between trials")
        flat = stimulus.reshape(stimulus.shape[0], -1).astype(np.float32)
        weights = response.astype(np.float32, copy=False)
        for lag in range(self.numerator.shape[1]):
            aligned_stimulus = flat[: stimulus.shape[0] - lag]
            aligned_response = weights[lag:]
            self.numerator[:, lag] += aligned_response.T @ aligned_stimulus
            self.spike_denominator[:, lag] += aligned_response.sum(axis=0)
            self.stimulus_sum[lag] += aligned_stimulus.sum(axis=0)
            self.frame_denominator[lag] += aligned_stimulus.shape[0]

    def finalize(self) -> np.ndarray:
        output = np.full_like(self.numerator, np.nan, dtype=np.float32)
        empirical_mean = self.stimulus_sum / self.frame_denominator[:, None]
        for cell in range(self.numerator.shape[0]):
            nonzero = self.spike_denominator[cell] > 0
            output[cell, nonzero] = (
                self.numerator[cell, nonzero]
                / self.spike_denominator[cell, nonzero, None]
                - empirical_mean[nonzero]
            )
        return output.reshape(
            self.numerator.shape[0], self.numerator.shape[1], self.height, self.width
        )

    def combined(self, other: StreamingSTA) -> StreamingSTA:
        if (
            self.numerator.shape != other.numerator.shape
            or self.height != other.height
            or self.width != other.width
        ):
            raise ValueError("STA accumulators have incompatible shapes")
        return StreamingSTA(
            self.numerator + other.numerator,
            self.spike_denominator + other.spike_denominator,
            self.stimulus_sum + other.stimulus_sum,
            self.frame_denominator + other.frame_denominator,
            self.height,
            self.width,
        )


__all__ = ["StreamingSTA"]
