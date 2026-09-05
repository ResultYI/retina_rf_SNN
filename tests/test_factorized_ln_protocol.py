from __future__ import annotations

from dataclasses import replace
import importlib

import pytest
import torch

from data.retinal_recording import RealSequenceSplit


def trial_split() -> RealSequenceSplit:
    generator = torch.Generator().manual_seed(31)
    events = (torch.rand(32, 150, 1, generator=generator) < 0.3).float()
    mask = torch.ones_like(events, dtype=torch.bool)
    mask[:, :30] = False
    return RealSequenceSplit(
        cone_drive=torch.randn(32, 150, 289, generator=generator),
        spike_counts=events.long(), spike_events=events, valid_mask=mask,
        source_image_ids=tuple(
            f"recording-live-frames-{segment*150:06d}-{(segment+1)*150-1:06d}-trial-{trial+1}"
            for segment in range(16) for trial in range(2)
        ),
        trial_indices=tuple(trial for _ in range(16) for trial in range(2)),
    )


def test_inner_dev_contract_exists() -> None:
    # Given the evaluation package, when resolving the inner split, then it exists.
    assert importlib.util.find_spec("evaluation.mechanistic_retina.factorized_ln_split") is not None


def test_trial_wise_80_20_guard_and_original_mask_are_preserved() -> None:
    # Given two interleaved trials, when splitting, then cuts use the full trial timeline.
    module = importlib.import_module("evaluation.mechanistic_retina.factorized_ln_split")
    source = trial_split()
    before = source.valid_mask.clone()
    result = module.make_inner_dev(source)
    assert torch.equal(source.valid_mask, before)
    assert int(result.train.valid_mask.sum()) == 2 * 1470
    assert int(result.development.valid_mask.sum()) == 2 * 390
    assert [(x.fit_stop, x.dev_start, x.trial_stop) for x in result.boundaries] == [(1860, 1920, 2400)] * 2
    assert result.train.cone_drive.shape[0] == 26
    assert result.development.cone_drive.shape[0] == 8
    for row, name in enumerate(result.development.source_image_ids):
        if "001800" in name:
            assert not bool(result.development.valid_mask[row, :120].any())
            assert bool(result.development.valid_mask[row, 120:].all())
            assert not bool(result.development.cone_drive[row, :60].any())
            assert not bool(result.development.spike_events[row, :60].any())


def test_inner_train_does_not_depend_on_dev_or_guard_targets() -> None:
    # Given changed guard/dev targets, when splitting, then inner training is identical.
    module = importlib.import_module("evaluation.mechanistic_retina.factorized_ln_split")
    source = trial_split()
    changed = source.spike_events.clone()
    changed[24:26, 60:] = 1 - changed[24:26, 60:]
    changed[26:] = 1 - changed[26:]
    original = module.make_inner_dev(source).train
    modified = module.make_inner_dev(replace(source, spike_events=changed, spike_counts=changed.long())).train
    assert torch.equal(original.spike_events, modified.spike_events)
    assert torch.equal(original.cone_drive, modified.cone_drive)
    assert torch.equal(original.valid_mask, modified.valid_mask)


def test_inner_dev_history_reset_excludes_fit_spikes() -> None:
    # Given modified fit spikes, when creating dev history, then dev features are identical.
    module = importlib.import_module("evaluation.mechanistic_retina.factorized_ln_split")
    model_module = importlib.import_module("baselines.center_surround_ln")
    source = trial_split()
    changed = source.spike_events.clone()
    changed[:24] = 1 - changed[:24]
    changed[24:26, :60] = 1 - changed[24:26, :60]
    before = module.make_inner_dev(source).development
    after = module.make_inner_dev(replace(source, spike_events=changed, spike_counts=changed.long())).development
    model = model_module.CenterSurroundLN(1000/150, 30, 61001)
    assert torch.equal(model.history_feature(before.spike_events), model.history_feature(after.spike_events))


def test_inner_dev_rejects_nonchronological_or_gapped_trial() -> None:
    # Given missing frame support within a trial, when splitting, then reject it.
    module = importlib.import_module("evaluation.mechanistic_retina.factorized_ln_split")
    source = trial_split()
    names = list(source.source_image_ids)
    names[2] = "recording-live-frames-000151-000300-trial-1"
    with pytest.raises(ValueError, match="contiguous"):
        module.make_inner_dev(replace(source, source_image_ids=tuple(names)))
