from __future__ import annotations

import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from contract import (  # noqa: E402
    ACTIVE,
    CONTRASTS,
    DT_MS,
    MACH_WIDTHS_PX,
    SBC_RADII_PX,
    build_bank,
)


def main() -> None:
    bank = build_bank()
    assert bank.drive.shape == (166, 150, 289)
    assert bank.history.shape == (166, 150, 1)
    assert torch.count_nonzero(bank.history) == 0
    assert int(ACTIVE.sum()) == 15
    assert DT_MS == 1000 / 150
    assert CONTRASTS == (0.0, 0.0625, 0.125, 0.25, 0.375, 0.5)
    assert SBC_RADII_PX == (0, 2, 4, 6, 8)
    assert MACH_WIDTHS_PX == (0, 2, 4, 8, 12, 16)

    sbc = [row for row in bank.comparisons if row.family == "SBC"]
    mach = [row for row in bank.comparisons if row.family == "Mach"]
    assert len(sbc) == len(CONTRASTS) * len(SBC_RADII_PX)
    assert len(mach) == 2 * len(CONTRASTS) * len(MACH_WIDTHS_PX)
    for row in sbc:
        bright, dark = bank.patches[row.a], bank.patches[row.b]
        assert torch.equal(bright[7:10, 7:10], torch.zeros(3, 3))
        assert torch.equal(dark[7:10, 7:10], torch.zeros(3, 3))
        if row.extent_px == 0 or row.contrast == 0:
            assert torch.equal(bright, dark)
            assert torch.count_nonzero(bright) == 0
    for row in mach:
        if row.contrast == 0:
            assert torch.equal(bank.patches[row.a], bank.patches[row.b])
    replay = torch.load(
        ROOT
        / "output/real_data/schottdorf_canonical_v1_shared_bc_frozen_applications_20260830/illusion/inputs.pt",
        weights_only=True,
    )
    sbc_original = next(row for row in sbc if row.contrast == 0.25 and row.extent_px == 8)
    mach_dark = next(
        row for row in mach
        if row.contrast == 0.25 and row.extent_px == 8 and row.signature.startswith("dark")
    )
    mach_bright = next(
        row for row in mach
        if row.contrast == 0.25 and row.extent_px == 8 and row.signature.startswith("bright")
    )
    for current, previous in (
        (sbc_original.a, 50), (sbc_original.b, 51),
        (mach_dark.a, 8), (mach_dark.b, 33),
        (mach_bright.a, 16), (mach_bright.b, 41),
    ):
        assert torch.equal(bank.drive[current], replay["cone_drive"][previous])
    active_drive = bank.drive[:, ACTIVE]
    assert torch.allclose(
        active_drive, bank.patches.flatten(1)[:, None].expand_as(active_drive), atol=1e-6, rtol=0
    )
    assert torch.count_nonzero(bank.drive[:, :45]) == 0
    assert torch.count_nonzero(bank.drive[:, 60:]) == 0
    print("PASS parametric stimulus and shared-zero-history contract")


if __name__ == "__main__":
    main()
