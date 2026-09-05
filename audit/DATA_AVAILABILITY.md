# Data and artifact availability

The handoff contains Git-visible source, configs, per-cell tables, trajectories,
final model checkpoints and selected RF/illusion tensors. Large original data
and duplicate cached inputs remain local; their paths, byte counts and SHA256
are listed in `cleanup_20260905/local_only_artifacts.csv`.

Removed superseded material is recoverable from the ZIP in
`cleanup_20260905/backup_verification.json`; exact original paths and hashes are
in `archived_files.csv`. The recovery ZIP is local, not a remote backup. Retain it
until obsolete material is no longer needed. No commit or push was performed.

## Schottdorf–Lee

Local paths: `data/real/schottdorf_lee_2021_repository/` and
`data/real/schottdorf_lee_2021_macaque/1x10_256.mpg`. Original recording hashes
and movie hashes are recorded in the final `run-manifest.json` and source
`results.json`. The materialized MPEG is 265,338,036 bytes, annex MD5
`d64bdae05eb07895a8f30cda287c5a74`. An annex pointer is not the data payload.
Official source URLs and acquisition synchronization evidence are in
the [frame-zero report](../.omo/evidence/schottdorf_lee_frame_zero_resolution.md).
The 750/751 issue remains unresolved; successful prediction does not settle it.

Training/natural-movie re-evaluation requires the materialized original data or
exact frozen inputs. Statistical result checks can use per-cell CSV/JSON.
Frozen synthetic/illusion tensors are selected for publication. Historical
absolute Windows paths map to the same relative suffix in a clone. Git HEAD
alone does not identify the dirty source used in a run; inspect its source hashes.

## Other data and software

Marmoset: G-Node DOI `10.12751/g-node.ejk8kx`, OpenRetina
`gollisch_lab/karamanlis_2024`. Retained RF centers and locality are a separate
historical marmoset lineage, not shared macaque geometry.

SC-adapted includes `baselines/spatial_contrast_LICENSE.txt` and pinned official
source snapshots in `.omo/evidence/spatial_contrast_baseline/sources/`.
Inherited LN filters and Bernoulli adaptation are explicit: it is neither the
faithful original SC implementation/experiment nor a model with only four
total parameter coordinates.

The Git publication manifest distinguishes current evidence from historical
dependencies. Excluded raw/large artifacts have not been silently deleted.
An external reviewer who cannot obtain them must label the affected independent
checks UNKNOWN rather than infer success from a report.
