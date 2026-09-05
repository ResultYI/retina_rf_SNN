from __future__ import annotations

from dataclasses import asdict
from io import BytesIO

import pytest
import torch
from torch import nn

from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticConfigError,
    MechanisticRetinaConfig,
)
from models.mechanistic_retina.model import (
    MechanisticGraphTemporalRetina,
    build_mechanistic_retina,
)


_CONTRACT = "bc-central-disk_ac-overlapping-full-disk"
_KEY = "_spatial_contract_id"


def _model() -> MechanisticGraphTemporalRetina:
    return build_mechanistic_retina(
        MechanisticRetinaConfig(
            architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
            cell_specific_gains=True,
        ),
        torch.tensor(((0.0, 0.0), (0.04, 0.0), (0.08, 0.0), (0.12, 0.0))),
        torch.zeros(1, 2),
        ("midget",),
        ("ON",),
    )


@pytest.mark.parametrize("strict", (True, False))
def test_old_spatial_state_is_rejected_before_parameter_mutation(strict: bool) -> None:
    # Given: the old state schema without a spatial identity, even with strict=False.
    model = _model()
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    old = {name: value.clone() for name, value in initial.items() if name != _KEY}
    old["rgc.response_bias"].add_(2)

    # When/Then: loading fails explicitly, before any tensor is overwritten.
    with pytest.raises(RuntimeError, match="spatial contract"):
        model.load_state_dict(old, strict=strict)
    assert all(torch.equal(value, initial[name]) for name, value in model.state_dict().items())


def test_config_and_checkpoint_preserve_new_spatial_identity_on_roundtrip() -> None:
    # Given: a new model serialized in memory, without training or disk checkpoints.
    model = _model()
    stream = BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = _model()

    # When: the complete state and explicit config identity are inspected.
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    snapshot = asdict(model.config)

    # Then: the public V1 config carries the spatial contract, not a new revision.
    assert snapshot.get("spatial_contract") == _CONTRACT
    assert torch.equal(model.state_dict()[_KEY], torch.tensor(list(_CONTRACT.encode()), dtype=torch.uint8))
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())


def test_wrong_spatial_identity_is_rejected_even_with_matching_tensor_shapes() -> None:
    # Given: a state with shape-compatible but mismatched spatial identity.
    model = _model()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    state[_KEY][0] = 0

    # When/Then: strict=False cannot bypass the semantic contract.
    with pytest.raises(RuntimeError, match="spatial contract"):
        model.load_state_dict(state, strict=False)


def test_nested_module_loading_cannot_bypass_spatial_identity() -> None:
    # Given: Canonical V1 nested in another module using recursive state loading.
    wrapper = nn.ModuleDict({"retina": _model()})
    old = {name: value for name, value in wrapper.state_dict().items()
           if name != f"retina.{_KEY}"}

    # When/Then: the spatial guard runs for recursive loading as well.
    with pytest.raises(RuntimeError, match="spatial contract"):
        wrapper.load_state_dict(old, strict=False)


def test_explicit_old_spatial_config_is_rejected() -> None:
    # Given/When/Then: a config explicitly requesting annular AC is unsupported.
    with pytest.raises(MechanisticConfigError, match="spatial contract"):
        MechanisticRetinaConfig(spatial_contract="bc-core_ac-exclusive-annulus")
