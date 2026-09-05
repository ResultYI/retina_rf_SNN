from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

import torch

from evaluation.v4_identity_endpoint import CellIdentityMetadata


class GroupEndpointKind(StrEnum):
    TYPE_COLLAPSED_POLARITY_LOCATION = "type_collapsed_polarity_paired_location"
    TYPE_POLARITY = "type_polarity"
    TYPE_POLARITY_LOCATION = "type_polarity_paired_location"
    POLARITY = "polarity"


@dataclass(frozen=True, slots=True)
class GroupSpec:
    groups: dict[str, tuple[int, ...]]
    keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroupScore:
    correct: float | None
    wrong_group: str | None
    wrong: float | None
    margin: float | None
    resolved: bool | None
    row: tuple[float | None, ...]


def group_spec(
    metadata: Sequence[CellIdentityMetadata], endpoint: GroupEndpointKind
) -> GroupSpec:
    rows: dict[str, list[int]] = {}
    for index, cell in enumerate(metadata):
        match endpoint:
            case GroupEndpointKind.TYPE_COLLAPSED_POLARITY_LOCATION:
                key = _location_key(cell, include_type=False)
            case GroupEndpointKind.TYPE_POLARITY:
                key = f"type={cell.type_id};polarity={cell.polarity}"
            case GroupEndpointKind.TYPE_POLARITY_LOCATION:
                key = _location_key(cell, include_type=True)
            case GroupEndpointKind.POLARITY:
                key = f"polarity={cell.polarity}"

        rows.setdefault(key, []).append(index)
    groups = {key: tuple(indices) for key, indices in rows.items()}
    return GroupSpec(groups, tuple(sorted(groups)))


def full_vectors(rf: torch.Tensor) -> torch.Tensor:
    return rf.movedim(2, 0).reshape(rf.shape[2], -1).double()


def score_groups(
    reference: torch.Tensor,
    query: torch.Tensor,
    spec: GroupSpec,
) -> tuple[GroupScore, ...]:
    centroids = {
        key: reference[list(indices)].mean(dim=0)
        for key, indices in spec.groups.items()
    }
    cell_groups = {
        index: key
        for key, indices in spec.groups.items()
        for index in indices
    }
    rows = []
    for index in range(reference.shape[0]):
        correct_key = cell_groups[index]
        remaining = tuple(
            value for value in spec.groups[correct_key] if value != index
        )
        if not remaining or len(spec.keys) < 2:
            rows.append(
                GroupScore(
                    None, None, None, None, None,
                    tuple(None for _ in spec.keys),
                )
            )
            continue
        values = [_cosine(query[index], centroids[key]) for key in spec.keys]
        correct = _cosine(query[index], reference[list(remaining)].mean(dim=0))
        values[spec.keys.index(correct_key)] = correct
        wrong = tuple(
            (value, key)
            for value, key in zip(values, spec.keys, strict=True)
            if key != correct_key
        )
        best, best_key = max(wrong)
        rows.append(
            GroupScore(
                correct,
                best_key,
                best,
                correct - best,
                correct > best,
                tuple(values),
            )
        )
    return tuple(rows)


def within_group_variance(vectors: torch.Tensor, spec: GroupSpec) -> float:
    values = []
    for indices in spec.groups.values():
        subset = vectors[list(indices)]
        values.extend(
            float(value)
            for value in (subset - subset.mean(dim=0)).square().sum(dim=1)
        )
    return sum(values) / len(values)


def between_group_distance(
    vectors: torch.Tensor, spec: GroupSpec
) -> float | None:
    centroids = [
        vectors[list(indices)].mean(dim=0) for indices in spec.groups.values()
    ]
    values = [
        1.0 - _cosine(left, right)
        for index, left in enumerate(centroids)
        for right in centroids[index + 1 :]
    ]
    return sum(values) / len(values) if values else None


def _location_key(cell: CellIdentityMetadata, *, include_type: bool) -> str:
    prefix = f"type={cell.type_id};" if include_type else ""
    return (
        f"{prefix}polarity={cell.polarity};replicate={cell.replicate_id};"
        f"position={cell.position_x:.9g},{cell.position_y:.9g}"
    )


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = (left.norm() * right.norm()).clamp_min(1e-12)
    return float(torch.dot(left, right) / denominator)


__all__ = [
    "GroupEndpointKind",
    "GroupScore",
    "GroupSpec",
    "between_group_distance",
    "full_vectors",
    "group_spec",
    "score_groups",
    "within_group_variance",
]
