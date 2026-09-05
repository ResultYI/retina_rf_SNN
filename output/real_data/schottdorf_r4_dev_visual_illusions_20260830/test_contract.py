from __future__ import annotations

import torch

from stimuli import PITCH_DEG, build_stimuli


def test_physical_matching_and_context_controls() -> None:
    bank = build_stimuli()
    assert bank.patches.shape == (63, 17, 17)
    for pair in bank.pairs:
        a, b = bank.patches[pair.a], bank.patches[pair.b]
        if pair.family == "Mach_bands":
            assert a[8, 8] == b[8, 8]
        else:
            assert torch.equal(a[7:10, 7:10], b[7:10, 7:10])
        if pair.control:
            assert torch.equal(a, b)
    assert bank.patches.min() >= -0.25
    assert bank.patches.max() <= 0.25
    assert PITCH_DEG == 4.6 / 256 * 3


def test_ramp_has_no_luminance_step_and_white_has_two_context_axes() -> None:
    bank = build_stimuli()
    ramp = bank.patches[12][8]
    assert torch.allclose(ramp[5:13] - ramp[4:12], torch.full((8,), 0.0625))
    assert torch.equal(ramp[:5], torch.full((5,), -0.25))
    assert torch.equal(ramp[12:], torch.full((5,), 0.25))
    white = bank.patches[bank.names.index("White_on_bright_bar")]
    assert white[8, 8] == 0
    assert white[4, 8] == 0.25
    assert white[8, 5] == -0.25


def test_pair_names_unique_and_no_baseline_in_pairs() -> None:
    bank = build_stimuli()
    assert len(set(bank.names)) == len(bank.names)
    assert torch.count_nonzero(bank.patches[-1]) == 0
    assert all(pair.a < 62 and pair.b < 62 for pair in bank.pairs)
