from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Final

import torch

OUT: Final = Path(__file__).resolve().parent
ROOT: Final = OUT.parents[2]
ORIGINAL: Final = OUT.parent / "schottdorf_r4_dev_visual_illusions_20260830"
sys.path.insert(0, str(ORIGINAL))
sys.path.insert(0, str(ROOT))

from stimuli import Pair, Stimuli, PITCH_DEG, build_stimuli
from metrics import Row, write_csv

VARIANTS: Final = (("White", "remote_contour_rearrangement", 58, 59, 4.0, math.pi / 2),
                   ("Hermann", "contour_rearrangement", 54, 55, 2.0, math.pi / 4))


def radius_squared() -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(8, -9, -1), torch.arange(-8, 9), indexing="ij")
    return x.square() + y.square()


def angular_permutation(patch: torch.Tensor, protected_radius: float, angle_radians: float) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(8, -9, -1), torch.arange(-8, 9), indexing="ij")
    r2 = radius_squared().flatten()
    angle = torch.atan2(y.float(), x.float()).flatten()
    result = patch.flatten().clone()
    for squared in r2.unique():
        radius = math.sqrt(float(squared))
        if radius <= protected_radius:
            continue
        indices = (r2 == squared).nonzero().flatten()
        ordered = indices[angle[indices].argsort()]
        phi = angle_radians * min(1.0, (radius - protected_radius) / 4.0)
        shift = math.floor(len(ordered) * phi / (2 * math.pi) + 0.5)
        result[ordered] = patch.flatten()[ordered.roll(shift)]
    return result.reshape(17, 17)


def diagnostic_bank() -> Stimuli:
    original = build_stimuli()
    saved = torch.load(ORIGINAL / "stimuli.pt", weights_only=True)
    assert torch.equal(original.patches, saved["patches"])
    names = list(original.names)
    patches = list(saved["patches"].unbind())
    pairs = list(original.pairs)
    for family, variant, a, b, radius, angle in VARIANTS:
        start = len(patches)
        for idx in (a, b, a + 2, b + 2):
            names.append(f"{family}_{variant}_{original.names[idx]}")
            patches.append(angular_permutation(saved["patches"][idx], radius, angle))
        pairs.append(Pair(family, f"{variant}_A_minus_B", start, start + 1))
        pairs.append(Pair(family, f"{variant}_control_A_minus_B", start + 2, start + 3, True))
    names.append("diagnostic_blank")
    patches.append(torch.zeros(17, 17))
    return Stimuli(tuple(names), torch.stack(patches), tuple(pairs), original.scenes, original.crop_centers_pixels)


def annulus_rows(bank: Stimuli) -> list[Row]:
    r2 = radius_squared()
    rings = [("disk_0_2px", r2 <= 4)] + [
        (f"annulus_{lo}_{hi}px", (r2 > lo * lo) & (r2 <= hi * hi))
        for lo, hi in ((2, 4), (4, 6), (6, 8), (8, math.sqrt(128)))
    ]
    rows: list[Row] = []
    for index, (family, variant, a, b, _, _) in enumerate(VARIANTS):
        for member, orig, diag in (("A", a, 63 + 4 * index), ("B", b, 64 + 4 * index),
                                  ("control_A", a + 2, 65 + 4 * index), ("control_B", b + 2, 66 + 4 * index)):
            for ring, mask in rings:
                base = bank.patches[orig][mask].double() + 1
                altered = bank.patches[diag][mask].double() + 1
                assert torch.equal(base.sort().values, altered.sort().values)
                rows.append({"family": family, "variant": variant, "target": member, "ring": ring,
                             "n_pixels": int(mask.sum()), "original_mean_LM": float(base.mean()),
                             "variant_mean_LM": float(altered.mean()), "mean_delta": float(altered.mean() - base.mean()),
                             "original_std_LM": float(base.std(correction=0)),
                             "variant_std_LM": float(altered.std(correction=0)),
                             "std_delta": float(altered.std(correction=0) - base.std(correction=0)),
                             "min_LM": float(altered.min()), "max_LM": float(altered.max()),
                             "fraction_dark": float((altered == 0.75).double().mean()),
                             "fraction_target_gray": float((altered == 1).double().mean()),
                             "fraction_bright": float((altered == 1.25).double().mean()),
                             "sorted_histogram_max_abs_error": float((base.sort().values - altered.sort().values).abs().max()),
                             "pixelwise_changed_count": int((base != altered).sum())})
    return rows


def prepare() -> None:
    assert not (OUT / "responses.pt").exists()
    bank = diagnostic_bank()
    original_input = torch.load(ORIGINAL / "input-tensors.pt", weights_only=True)
    diag_drive = bank.patches[63:].flatten(1)[:, None] * original_input["envelope"][None, :, None]
    torch.save({"patches": bank.patches, "names": bank.names, "original_drive": original_input["cone_drive"],
                "original_history": original_input["history"], "diagnostic_drive": diag_drive,
                "diagnostic_history": torch.zeros(9, 150, 1), "time_ms": original_input["time_ms"],
                "envelope": original_input["envelope"]}, OUT / "inputs.pt")
    stats = annulus_rows(bank)
    write_csv(OUT / "annular-luminance.csv", stats)
    (OUT / "protocol.json").write_text(json.dumps({
        "original_source": str(ORIGINAL), "original_contract_unchanged": True,
        "input": "Identical saved 17x17 relative L+M Weber drive; native dt=1000/150 ms; exact original envelope and zero observed-spike history",
        "variant_scope": "One fixed custom diagnostic per family, not exact paper reproductions; no perceptual effect asserted; no stimulus search or model-output-dependent construction",
        "White_motivation": "Change remote contour continuity/orientation while preserving target and near-junction neighborhood; motivated by Howe 2001/2005 contextual variants with unchanged local junctions",
        "White_sources": ["https://doi.org/10.1068/p3212", "https://doi.org/10.1068/p5414"],
        "Hermann_motivation": "Disrupt straight alley continuity without changing intersection target, motivated by Geier et al. 2008 straightness manipulation. At 17x17 resolution this produces fragmented contours, not a smooth curved grid.",
        "Hermann_source": "https://doi.org/10.1068/p5622",
        "construction": "At each exact squared pixel radius, sort pixels by atan2(y,x). Cyclically permute luminance by floor(n*phi(r)/(2*pi)+0.5). phi(r)=phi_max*clip((r-r_protected)/4,0,1). Values at r<=r_protected unchanged. This is lattice-shell rearrangement, not an exact continuous rotation.",
        "variants": [{"family": f, "variant": v, "original_A": a, "original_B": b,
                      "protected_radius_pixels": r, "protected_radius_deg": r * PITCH_DEG,
                      "max_angle_radians": angle} for f, v, a, b, r, angle in VARIANTS],
        "local_views": "A and B are separately centered local diagnostic views, not crops claimed to belong to one globally warped scene",
        "annulus_definition": "Target-centered Euclidean pixel radius; ring endpoints (0,2],(2,4],(4,6],(6,8],(8,sqrt(128)] with origin included in first disk; pitch=0.05390625 deg. Relative L+M=1+drive.",
        "matching_boundary": "Original-to-variant histogram at every exact target-centered radius is preserved separately for A and B, including target. A and B surrounds are not claimed equal. Off-center LN Gaussian-weighted luminance is not claimed matched.",
        "metrics": "Unmodified prior response_metrics, cell_rows, mach_rows, aggregate. Original-vs-variant and LN-vs-R4 are signed differences of response time courses, measured using those same metrics.",
        "pairs": [asdict(pair) for pair in bank.pairs], "protocol_frozen_before_model_inference": True,
        "max_annular_mean_delta": max(abs(float(r["mean_delta"])) for r in stats),
        "max_annular_std_delta": max(abs(float(r["std_delta"])) for r in stats),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    prepare()
