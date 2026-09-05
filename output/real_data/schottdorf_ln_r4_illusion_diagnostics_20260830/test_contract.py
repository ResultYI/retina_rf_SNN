from __future__ import annotations

import math
from pathlib import Path

import torch

from diagnostic_stimuli import angular_permutation

SOURCE = Path(__file__).resolve().parent.parent / "schottdorf_r4_dev_visual_illusions_20260830"


def test_each_radius_histogram_is_preserved() -> None:
    patch = torch.load(SOURCE / "stimuli.pt", weights_only=True)["patches"][54]
    transformed = angular_permutation(patch, 2.0, math.pi / 4)
    y, x = torch.meshgrid(torch.arange(-8, 9), torch.arange(-8, 9), indexing="ij")
    radius2 = x.square() + y.square()
    for ring in radius2.unique():
        assert torch.equal(patch[radius2 == ring].sort().values,
                           transformed[radius2 == ring].sort().values)


def test_white_target_and_near_context_are_unchanged() -> None:
    patch = torch.load(SOURCE / "stimuli.pt", weights_only=True)["patches"][58]
    transformed = angular_permutation(patch, 4.0, math.pi / 2)
    y, x = torch.meshgrid(torch.arange(-8, 9), torch.arange(-8, 9), indexing="ij")
    protected = x.square() + y.square() <= 16
    assert torch.equal(transformed[protected], patch[protected])
    assert torch.count_nonzero(transformed - patch) > 0


def test_hermann_intersection_and_corridor_targets_are_matched() -> None:
    patches = torch.load(SOURCE / "stimuli.pt", weights_only=True)["patches"]
    a = angular_permutation(patches[54], 2.0, math.pi / 4)
    b = angular_permutation(patches[55], 2.0, math.pi / 4)
    assert torch.equal(a[7:10, 7:10], b[7:10, 7:10])
    assert torch.equal(a[7:10, 7:10], patches[54, 7:10, 7:10])
    assert torch.count_nonzero(a - patches[54]) > 0
