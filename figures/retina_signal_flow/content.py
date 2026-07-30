from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class CardContent:
    title: str
    lines: tuple[str, ...]


CONE_RESPONSE: Final = CardContent(
    "CONE RESPONSE TIMING",
    (
        "HUMAN · in vivo paired-flash ERG estimate",
        "Dim-flash Tpeak ≈ 20 ms",
        "[van Hateren & Lamb, 2006]",
        "MACAQUE · direct cone voltage responses",
        "5,000 R*/cone/s",
        "L 36.7 ms · M 36.3 ms · S 45.3 ms",
        "50,000 R*/cone/s",
        "L 33.1 ms · M 34.1 ms · S 44.5 ms",
        "[Baudin et al., 2019]",
    ),
)

CONE_DENSITY: Final = CardContent(
    "CONE DENSITY & COMPOSITION",
    (
        "HUMAN",
        "Peak 199,000 cones/mm²; total 4.6 M",
        "[Curcio et al., 1990]",
        "L:M 1.1:1–16.5:1",
        "[Hofer et al., 2005]",
        "MACAQUE",
        "M. nemestrina: peak 210,000/mm²; total 3.1 M",
        "[Packer et al., 1989]",
        "M. mulatta: L:M 1.03:1 ± 0.02 (proxy)",
        "[Munds et al., 2022]",
    ),
)

H1_DATA: Final = CardContent(
    "H1 HORIZONTAL CELL",
    (
        "HUMAN · direct physiological RF: ?",
        "MACAQUE · combined RF diameter",
        "122 µm at 4 mm; 309 µm at 11 mm",
        "narrow direct field + broad coupled field",
        "[Packer & Dacey, 2002]",
        "L/M input; no measurable S input",
        "[Dacey et al., 1996]",
        "Timing · HUMAN: ?",
        "Timing · MACAQUE",
        "ΔTpeak (H1 − cone) ≈ 4 ms",
        "[Smith et al., 2001]",
    ),
)

MIDGET_BIPOLAR: Final = CardContent(
    "MIDGET BIPOLAR",
    (
        "HUMAN · direct physiological RF: ?",
        "MACAQUE · peripheral in vitro",
        "center 42 µm (31–51)",
        "surround 467 µm (432–515)",
        "diameter = 2 × Gaussian radius",
        "[Dacey et al., 2000]",
        "Physiological timing · HUMAN: ?",
        "Physiological timing · MACAQUE: ?",
    ),
)

DIFFUSE_BIPOLAR: Final = CardContent(
    "DIFFUSE BIPOLAR",
    (
        "HUMAN · direct physiological RF: ?",
        "MACAQUE · peripheral in vitro",
        "center 92 µm (74–114)",
        "surround 743 µm (594–1048)",
        "diameter = 2 × Gaussian radius",
        "[Dacey et al., 2000]",
        "Physiological timing · HUMAN: ?",
        "Physiological timing · MACAQUE: ?",
    ),
)

AMACRINE_DATA: Final = CardContent(
    "LOCAL AMACRINE PATHWAY",
    (
        "Timing · HUMAN: ?",
        "Timing · MACAQUE: ?",
        "Subtype required",
        "No universal biological delay",
    ),
)

RGC_HUMAN: Final = CardContent(
    "RGC · HUMAN",
    (
        "ANATOMY, not physiological RF",
        "Midget D = 8.64 E^1.04",
        "Parasol D = 70.2 E^0.65",
        "D in µm; E in mm",
        "ON dendritic fields 30–50% larger",
        "[Dacey & Petersen, 1992]",
        "Physiological response timing: ?",
    ),
)

RGC_MACAQUE: Final = CardContent(
    "RGC · MACAQUE",
    (
        "P/M center–surround depends on eccentricity",
        "[Croner & Kaplan, 1995]",
        "Midget STA Tpeak: 67 ± 6 ms foveal",
        "53 ± 6 ms central; 37 ± 4 ms peripheral",
        "[Sinha et al., 2017]",
        "ON parasol impulse timing is 10–20% shorter",
        "than OFF for peak, trough, and zero crossing",
        "[Chichilnisky & Kalmar, 2002]",
    ),
)
