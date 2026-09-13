# Phase R1 model comparison — NOT REACHED

Official reliability preflight is UNVERIFIED. No model evaluation protocol, interpolation choice, numeric metric or decision threshold was operationalized or tested against model results.

The authorized primary contract remains `STIMULUS_ONLY_ZERO_HISTORY`: identical visual stimulus and zero observed-spike history. Any later authorized secondary `FORMAL_CONDITIONAL_HISTORY` computation must be separate and must not receive primary reliability normalization. No such predictions were generated here.

The existing model loader uses 150 Hz and `_LIVE_START_FRAME=751`; raw reliability uses 1 ms bins. No frame-zero change, temporal shift search, interpolation, smoothing or timebase conversion was performed. The author states the repeated stimulus is the first minute of the 10-minute movie. Acquisition-corrected spike times are already relative to live video; the internal `Video Starts` value must not be subtracted again.

Aligned checkpoint identity was resolved from the requested fixed-alignment audit's `analysis-manifest.json` and cross-checked against its `artifact-manifest.json` (that audit does not use the literal filename `evidence_manifest.json`). LN/CNN frozen identities were recovered from the existing held-out evidence manifest. All 66 file hashes match. No checkpoint tensors were loaded and no model was run.

The 20 available repeated recordings are included in the aligned training-contract recording IDs. Thus these data cannot be advertised as an untouched predictive test. Train/dev temporal usage and a common scored time mask would require completion of this gated model protocol; no new predictive claim is made.
