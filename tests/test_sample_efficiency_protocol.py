from __future__ import annotations

from pathlib import Path

import pytest
import torch

from evaluation.mechanistic_retina.rf_base import load_candidate0
from evaluation.mechanistic_retina.spike_banks import (
    generate_nested_spike_bank,
    slice_spike_bank,
    tensor_sha256,
)
from evaluation.model_comparison.sample_efficiency_protocol import (
    SampleEfficiencyProtocolError,
    build_sample_efficiency_slices,
    load_sample_efficiency_protocol,
)
from training.mechanistic_retina.stages import build_seed_data


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/model_comparison_sample_efficiency_t2.yaml"
EXPECTED_25_INDICES = (3, 8, 9, 13, 15, 16, 21, 25, 28, 31, 38, 52, 55, 62, 63, 66, 69, 75, 76, 87, 88, 93, 98, 99, 101, 102, 107, 111)
EXPECTED_50_INDICES = (
    2,
    3,
    4,
    7,
    8,
    9,
    11,
    13,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    25,
    26,
    28,
    30,
    31,
    32,
    36,
    38,
    39,
    42,
    52,
    55,
    56,
    57,
    59,
    61,
    62,
    63,
    64,
    65,
    66,
    68,
    69,
    74,
    75,
    76,
    85,
    87,
    88,
    90,
    92,
    93,
    95,
    96,
    98,
    99,
    101,
    102,
    107,
    111,
)
EXPECTED_25_HASHES = ("75b3a3f66664750164225c46181c59ff5cd1b7dfffe05718c3d0e9c9e0a23d1e", "ebf686cb06da9fec98101a5376d6516e3981c89841b92198c71a24d9c685ea17", "70192929c309256ee71bf0508b65d8bc210611885562ad0df016c386a6cfb1d8")
EXPECTED_50_HASHES = ("e14a67532b479ec0798261c0c885d45a708951978bb45f611d132c4a7e1606ee", "9d74734107ba2b65c1c575d5a94d3522260a108e9b95f631793c762b040013c1", "3945f99d6b30c8e6e0a65176e534e66e15ba7680a4b0dacfea66b3f0154347df")
EXPECTED_25_BANK_HASHES = (
    "565051fa201900a324a468b1d6216dc4b51780e07f6b9e618712796eac68b083",
    "4463607e48104fc6eb6e8be07ee9e7f9a813ec5c59b6002107d5ff9e66bef42d",
    "b998f08e410d8bc93da46fb3da1bc84ea8570ed2ce0dff42b99556ddc588b9cb",
)
EXPECTED_50_BANK_HASHES = (
    "0bdd2e8d75008135b83c0d60f338be077d22a1e774db9dd99756042878b3d491",
    "3de11d10b6734dadb80f7c17ba7a4c7f26b7dcabcc358f19835fd95aeaec7f50",
    "b901f798091bc92202f5e5283a105979eb3947a9fbbf696dfe1a831fd314fa8f",
)


def test_protocol_counts_are_nested_and_sorted_when_loaded() -> None:
    # Given: the frozen sample-efficiency config.
    # When: the protocol is parsed at the boundary.
    protocol = load_sample_efficiency_protocol(CONFIG)

    # Then: the canonical identity and nested stimulus membership are fixed.
    assert protocol.fractions == (0.25, 0.5, 1.0)
    assert protocol.canonical_config.data_seed == 19
    assert protocol.canonical_config.bank_seeds == (31001, 31002, 31003)
    assert protocol.canonical_config.model_seeds == (19, 20, 21)
    assert protocol.canonical_config.trials == 2
    assert protocol.canonical_config.steps == 400
    assert tuple(item.train_count for item in protocol.subsets) == (28, 56, 112)
    assert protocol.subsets[0].indices == tuple(sorted(protocol.subsets[0].indices))
    assert protocol.subsets[1].indices == tuple(sorted(protocol.subsets[1].indices))
    assert set(protocol.subsets[0].indices) < set(protocol.subsets[1].indices)
    assert set(protocol.subsets[1].indices) < set(protocol.subsets[2].indices)
    assert protocol.subsets[0].indices == EXPECTED_25_INDICES
    assert protocol.subsets[1].indices == EXPECTED_50_INDICES
    assert protocol.subsets[2].indices == tuple(range(112))


def test_slices_preserve_canonical_identity_when_using_100_percent() -> None:
    # Given: canonical Candidate0 data and spike banks.
    protocol = load_sample_efficiency_protocol(CONFIG)
    candidate = load_candidate0(
        ROOT / protocol.canonical_config.candidate0_path,
        usage=protocol.canonical_config.candidate_teacher_usage,
        reference_candidate_index=(
            protocol.canonical_config.candidate_teacher_reference_index
        ),
    )
    data = build_seed_data(protocol.canonical_config.data_seed, candidate)
    banks = tuple(
        slice_spike_bank(
            generate_nested_spike_bank(
                data.train_probability[:, 0],
                data.validation_probability[:, 0],
                seed=seed,
                max_trials=64,
            ),
            protocol.canonical_config.trials,
        )
        for seed in protocol.canonical_config.bank_seeds
    )

    # When: all requested fraction slices are built.
    slices = build_sample_efficiency_slices(protocol, data, banks)

    # Then: train slices use sorted shared indices and validation remains unchanged.
    assert tuple(item.train_count for item in slices) == (28, 56, 112)
    assert slices[-1].train_cone_sha256 == tensor_sha256(data.train_cones)
    assert slices[-1].train_probability_sha256 == tensor_sha256(data.train_probability)
    assert slices[-1].train_mask_sha256 == tensor_sha256(data.train_mask)
    assert slices[-1].validation_cone_sha256 == tensor_sha256(data.validation_cones)
    assert slices[-1].validation_probability_sha256 == tensor_sha256(
        data.validation_probability
    )
    assert slices[-1].validation_mask_sha256 == tensor_sha256(data.validation_mask)
    assert slices[0].validation_cones is data.validation_cones
    assert slices[1].validation_probability is data.validation_probability
    assert slices[2].validation_mask is data.validation_mask
    assert len({sample_slice.validation_probability_sha256 for sample_slice in slices}) == 1
    assert (
        slices[0].train_cone_sha256,
        slices[0].train_probability_sha256,
        slices[0].train_mask_sha256,
    ) == EXPECTED_25_HASHES
    assert (
        slices[1].train_cone_sha256,
        slices[1].train_probability_sha256,
        slices[1].train_mask_sha256,
    ) == EXPECTED_50_HASHES
    assert tuple(bank.train_sha256 for bank in slices[0].banks) == EXPECTED_25_BANK_HASHES
    assert tuple(bank.train_sha256 for bank in slices[1].banks) == EXPECTED_50_BANK_HASHES
    for sample_slice in slices:
        index = torch.tensor(sample_slice.indices, dtype=torch.long)
        assert torch.equal(sample_slice.train_cones, data.train_cones[index])
        assert torch.equal(sample_slice.train_probability, data.train_probability[index])
        assert torch.equal(sample_slice.train_mask, data.train_mask[index])
        assert sample_slice.validation_cone_sha256 == slices[-1].validation_cone_sha256
        assert sample_slice.validation_mask_sha256 == slices[-1].validation_mask_sha256
        for bank_slice in sample_slice.banks:
            source_bank = next(bank for bank in banks if bank.seed == bank_slice.seed)
            assert torch.equal(bank_slice.train_spikes, source_bank.train_spikes[index])
            assert bank_slice.validation_spikes is source_bank.validation_spikes
            assert bank_slice.validation_sha256 == source_bank.validation_sha256
            if sample_slice.fraction == 1.0:
                expected = protocol.canonical_config.bank_hashes[bank_slice.seed]
                assert (bank_slice.train_sha256, bank_slice.validation_sha256) == expected


def test_loader_fails_with_typed_error_when_config_is_malformed(tmp_path: Path) -> None:
    # Given: malformed sample-efficiency boundary files.
    duplicate_fraction = tmp_path / "duplicate.json"
    duplicate_fraction.write_text(
        """{
  "canonical_config_path": "configs/model_comparison_t2.yaml",
  "output_dir": ".omo/evidence/sample-efficiency-active-dof-t2",
  "run_dir": "runs/sample_efficiency_active_dof_t2",
  "selection_seed": 19,
  "fractions": [0.25, 0.25, 1.0]
}""",
        encoding="utf-8",
    )
    wrong_seed = tmp_path / "wrong-seed.json"
    wrong_seed.write_text(
        """{
  "canonical_config_path": "configs/model_comparison_t2.yaml",
  "output_dir": ".omo/evidence/sample-efficiency-active-dof-t2",
  "run_dir": "runs/sample_efficiency_active_dof_t2",
  "selection_seed": 20,
  "fractions": [0.25, 0.5, 1.0]
}""",
        encoding="utf-8",
    )
    missing_config = tmp_path / "missing-config.json"
    missing_config.write_text(
        """{
  "canonical_config_path": "configs/does-not-exist.yaml",
  "output_dir": ".omo/evidence/sample-efficiency-active-dof-t2",
  "run_dir": "runs/sample_efficiency_active_dof_t2",
  "selection_seed": 19,
  "fractions": [0.25, 0.5, 1.0]
}""",
        encoding="utf-8",
    )
    wrong_fraction_set = tmp_path / "wrong-fraction-set.json"
    wrong_fraction_set.write_text(
        """{
  "canonical_config_path": "configs/model_comparison_t2.yaml",
  "output_dir": ".omo/evidence/sample-efficiency-active-dof-t2",
  "run_dir": "runs/sample_efficiency_active_dof_t2",
  "selection_seed": 19,
  "fractions": [0.25, 0.75, 1.0]
}""",
        encoding="utf-8",
    )

    # When/Then: parsing fails before any training surface is needed.
    with pytest.raises(SampleEfficiencyProtocolError, match="FRACTIONS_NOT_UNIQUE"):
        load_sample_efficiency_protocol(duplicate_fraction)
    with pytest.raises(SampleEfficiencyProtocolError, match="SELECTION_SEED_NOT_19"):
        load_sample_efficiency_protocol(wrong_seed)
    with pytest.raises(SampleEfficiencyProtocolError, match="CANONICAL_CONFIG_MISSING"):
        load_sample_efficiency_protocol(missing_config)
    with pytest.raises(SampleEfficiencyProtocolError, match="FRACTIONS_NOT_FROZEN"):
        load_sample_efficiency_protocol(wrong_fraction_set)
