from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Literal, assert_never

import pytest
import torch

from evaluation.mechanistic_retina.karamanlis_v1_rf_validation import (
    RFValidationError,
    validate_v1_checkpoint,
)
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MECHANISTIC_MODEL_REVISION,
    MechanisticConfigError,
    MechanisticRetinaConfig,
)
from models.mechanistic_retina.model import (
    MechanisticGraphTemporalRetina,
    build_mechanistic_retina,
)
from models.mechanistic_retina.pathway_spatial_geometry import PathwaySpatialGeometry


def _config() -> MechanisticRetinaConfig:
    return MechanisticRetinaConfig(
        architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
        cell_specific_gains=True,
    )


def _positions() -> torch.Tensor:
    return torch.tensor(((0.0, 0.0), (0.04, 0.0), (0.12, 0.0), (0.16, 0.0)))


def _model() -> MechanisticGraphTemporalRetina:
    return build_mechanistic_retina(
        _config(), _positions(), torch.zeros(1, 2), ("midget",), ("ON",),
    )


@pytest.mark.parametrize("supports", (
    ((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 1.0, 0.0)),
    ((1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 0.0, 1.0)),
    ((1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
    ((1.0, 0.5, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0)),
))
def test_custom_geometry_rejected_when_masks_are_not_exact_full_disks(
    supports: tuple[tuple[float, ...], tuple[float, ...]],
) -> None:
    # Given: the audit interior-hole counterexample and other non-disk masks.
    geometry = PathwaySpatialGeometry(
        torch.ones(1, 2, 4), torch.tensor([supports[0]]), torch.tensor([supports[1]]),
    )
    # When/Then: incompatible geometry is rejected during construction.
    with pytest.raises(ValueError, match="support|disk"):
        build_mechanistic_retina(
            _config(), _positions(), torch.zeros(1, 2), ("midget",), ("ON",),
            pathway_spatial_geometry=geometry,
        )


def test_custom_geometry_preserved_when_masks_equal_full_disks() -> None:
    # Given: valid full disks and an explicitly supplied nonnegative spatial basis.
    geometry = PathwaySpatialGeometry(
        torch.tensor([[[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]]]),
        torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
    )
    # When: the current Canonical builder accepts a conforming custom geometry.
    model = build_mechanistic_retina(
        _config(), _positions(), torch.zeros(1, 2), ("midget",), ("ON",),
        pathway_spatial_geometry=geometry,
    )
    # Then: geometry values are preserved without any radius or basis adjustment.
    assert torch.equal(model.feature_bank.spatial_basis, geometry.spatial_basis)
    assert torch.equal(model.feature_bank.bc_support, geometry.bc_support)
    assert torch.equal(model.feature_bank.ac_support, geometry.ac_support)


@pytest.mark.parametrize("invalid", (float("nan"), float("inf")))
def test_geometry_rejected_when_a_cone_position_is_nonfinite(invalid: float) -> None:
    # Given: an invalid cone outside otherwise populated BC and AC disks.
    positions = _positions()
    positions[-1, 0] = invalid
    # When/Then: nonfinite coordinates cannot silently become masked-out cones.
    with pytest.raises(ValueError, match="finite"):
        build_mechanistic_retina(
            _config(), positions, torch.zeros(1, 2), ("midget",), ("ON",),
        )


@pytest.mark.parametrize("direct_constructor", (False, True))
def test_canonical_entry_rejects_legacy_configuration(direct_constructor: bool) -> None:
    # Given: the explicit legacy configuration accepted in the original audit.
    constructor = MechanisticGraphTemporalRetina if direct_constructor else build_mechanistic_retina
    legacy = MechanisticRetinaConfig(architecture_mode=ArchitectureMode.LEGACY)
    # When/Then: neither public construction route may execute legacy as Canonical V1.
    with pytest.raises(MechanisticConfigError, match="mechanism_identifiable"):
        constructor(legacy, _positions(), torch.zeros(1, 2), ("midget",), ("ON",))


@pytest.mark.parametrize("mutation", (
    "legacy", "missing_mode", "causal", "spatial", "missing_config_field",
    "revision", "missing_revision", "schema", "missing_causal_state",
))
def test_canonical_validator_rejects_incompatible_checkpoint_identity(
    mutation: Literal[
        "legacy", "missing_mode", "causal", "spatial", "missing_config_field",
        "revision", "missing_revision", "schema", "missing_causal_state",
    ],
) -> None:
    # Given: a full current payload, then one incompatible serialized identity.
    model = _model()
    config = asdict(model.config)
    checkpoint = {
        "schema": "karamanlis_2024_marmoset_rf_geometry_canonical_retina_v1",
        "revision": MECHANISTIC_MODEL_REVISION,
        "stage": "best_trained",
        "model_config": config,
        "model": deepcopy(model.state_dict()),
    }
    match mutation:
        case "legacy":
            config["architecture_mode"] = "legacy"
        case "missing_mode":
            del config["architecture_mode"]
        case "causal":
            config["causal_contract"] = "independent-ac"
        case "spatial":
            config["spatial_contract"] = "exclusive-annulus"
        case "missing_config_field":
            del config["bc_delay_bounds_ms"]
        case "revision":
            checkpoint["revision"] = MECHANISTIC_MODEL_REVISION - 1
        case "missing_revision":
            del checkpoint["revision"]
        case "schema":
            checkpoint["schema"] = "legacy"
        case "missing_causal_state":
            del checkpoint["model"]["_causal_contract_id"]
        case unreachable:
            assert_never(unreachable)
    # When/Then: strict loading is never reached for an incompatible V1 payload.
    with pytest.raises(RFValidationError):
        validate_v1_checkpoint(checkpoint)


@pytest.mark.parametrize("strict", (False, True))
def test_legacy_state_rejected_before_mutation_regardless_of_strict(strict: bool) -> None:
    # Given: the original legacy state format lacks both canonical identity markers.
    model = _model()
    before = deepcopy(model.state_dict())
    legacy = {name: value.clone() for name, value in before.items()
              if name not in {"_causal_contract_id", "_spatial_contract_id"}}
    legacy["rgc.response_bias"].add_(1.0)
    # When/Then: semantic rejection does not depend on PyTorch key strictness.
    with pytest.raises(RuntimeError, match="causal contract"):
        model.load_state_dict(legacy, strict=strict)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_canonical_validator_accepts_current_complete_identity() -> None:
    # Given: the actual current schema, config fields, and contract state markers.
    model = _model()
    checkpoint = {
        "schema": "karamanlis_2024_marmoset_rf_geometry_canonical_retina_v1",
        "revision": MECHANISTIC_MODEL_REVISION,
        "stage": "best_trained",
        "model_config": asdict(model.config),
        "model": model.state_dict(),
    }
    # When/Then: full current identity remains admissible without conversion.
    validate_v1_checkpoint(checkpoint)


@pytest.mark.parametrize("strict", (False, True))
@pytest.mark.parametrize("invalid_geometry", ("hole", "basis_hole", "basis_outside", "missing_basis"))
def test_checkpoint_geometry_rejected_before_state_mutation(
    strict: bool, invalid_geometry: Literal["hole", "basis_hole", "basis_outside", "missing_basis"],
) -> None:
    # Given: an incompatible geometry state carrying both current contract markers.
    model = _model()
    before = deepcopy(model.state_dict())
    incoming = deepcopy(before)
    match invalid_geometry:
        case "hole":
            incoming["feature_bank.bc_support"][0, 1] = 0
            incoming["feature_bank.ac_support"][0, 1] = 0
            incoming["feature_bank.path_spatial_basis"][:, :, :, 1] = 0
        case "basis_outside":
            incoming["feature_bank.path_spatial_basis"][:, :, :, -1] = 0.2
        case "basis_hole":
            incoming["feature_bank.path_spatial_basis"][:, :, :, 1] = 0
        case "missing_basis":
            del incoming["feature_bank.path_spatial_basis"]
        case unreachable:
            assert_never(unreachable)
    incoming["rgc.response_bias"].add_(2.0)
    # When/Then: rejection occurs before any checkpoint tensor is loaded.
    with pytest.raises(RuntimeError, match="spatial|geometry"):
        model.load_state_dict(incoming, strict=strict)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_spatial_checkpoint_identity_preserves_dtype_converted_roundtrip() -> None:
    # Given: buffers computed in float32 then promoted by the standard module API.
    model = _model().double()
    restored = _model().double()
    # When: a complete converted state is strict-loaded without recomputing geometry.
    restored.load_state_dict(model.state_dict(), strict=True)
    # Then: valid dtype conversion retains exact saved buffer values.
    assert all(torch.equal(value, restored.state_dict()[name])
               for name, value in model.state_dict().items())
