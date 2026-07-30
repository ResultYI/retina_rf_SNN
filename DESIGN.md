# Retina RF SNN Figure Design System

## 1. Atmosphere & Identity

Structured scientific flow diagram. Five numbered stage bands establish the
reading order and a compact central circuit carries the biology. Quantitative
evidence is kept in the accompanying report tables instead of being repeated
under the circuit. The visual language uses precise geometry, restrained
tinting, and thin technical connectors rather than decorative anatomy or
simulated cone mosaics.

## 2. Color

| Role | Token | Value | Usage |
|---|---|---:|---|
| Canvas | `canvas` | `#FDFDFC` | Figure background |
| Surface | `surface` | `#FFFFFF` | Main cards |
| Text | `ink` | `#17181B` | Primary text and connections |
| Muted text | `muted` | `#5F6269` | Secondary labels |
| Border | `border` | `#A7A9AE` | Cards and dividers |
| Input stage | `input` | `#666A70` | Input header and modules |
| Cone stage | `cone_stage` | `#78906B` | Cone header and evidence |
| Processing stage | `processing` | `#6D568D` | H1 and processing header |
| Amacrine stage | `amacrine` | `#9B6A1E` | Amacrine header |
| Output stage | `output` | `#2C6294` | RGC and output header |
| L cone | `cone_l` | `#B64C52` | L-cone symbol |
| M cone | `cone_m` | `#59834F` | M-cone symbol |
| S cone | `cone_s` | `#3F6FA8` | S-cone symbol |
| ON pathway | `on` | `#C27A12` | ON connections and RF center |
| OFF pathway | `off` | `#2F67A3` | OFF connections and RF center |
| Inhibitory circuit | `inhibitory` | `#5B3D77` | H1 and amacrine circuits |

Colors encode cell/pathway identity only. Large areas remain white or neutral.

## 3. Typography

| Level | Size | Weight | Usage |
|---|---:|---:|---|
| Display | 34 pt | 700 | Figure title |
| Stage | 13 pt | 700 | Numbered stage bands |
| Node | 12 pt | 600 | Cell and module labels |
| Body | 8.5 pt | 400 | Mechanism labels |
| Caption | 7.5 pt | 400 | Secondary annotations |

Font stack: `Arial`, then `DejaVu Sans`, then generic sans-serif. SVG text stays
editable.

## 4. Spacing & Layout

The canvas is 4800 x 1980 px at 300 dpi in a wide scientific-flow ratio.
Spacing uses a 10 px base unit. Five stage columns are separated by dotted
guides. Signal flow occupies the main field; one compact connection legend
anchors the footer. Numerical physiology and citations live in report tables.

## 5. Components

### Stage band

- Numbered, rounded rectangle with a stage-specific pale fill.
- Thin colored outline; no shadow.
- Width follows the biological content beneath it.

### Biological node

- Geometric silhouette or compact cell panel plus a short label.
- Pale stage tint, 1.2 pt outline, no decorative anatomy.
- ON and OFF variants use amber and blue and align vertically.

### Receptive-field tile

- Small bordered tile beside the owning bipolar or RGC node.
- Two concentric circles encode center and surround.
- ON uses an amber center; OFF uses a blue center.

### Connection

- Solid arrow: direct feedforward excitation.
- Curved arrow: feedback.
- T-bar: inhibition.
- Dashed line: weak or absent input under the cited measurement.

## 6. Motion & Interaction

Not applicable. The artifact is a static scientific figure.

## 7. Depth & Surface

Thin borders plus subtle stage-tinted fills. No gradients, drop shadows,
textures, glass effects, or 3D treatment.
