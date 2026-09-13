# Range-selection rule, recorded before new predictions

The requested primary candidate is live time [20,60) s, live frames [3000,9000) at 150 Hz. Keep the existing decoded-frame start 751 and all production binning, sequence, warmup, reset and history semantics. Frame-zero acquisition provenance is not resolved by this task.

Eligibility requires all 22 cells and all 37 recordings, with each recording's actual independent movie trials, complete stimulus/target timeline support, and no identified use of the candidate responses/predictions/metrics for model development. File hashing, duration/header validation and raw-timestamp parsing for source validation are distinguished from training or performance inspection.

If the primary range is not eligible, inspect candidate starts in ascending whole seconds, retaining a fixed 40-second duration and all recordings/trials. Reject any candidate overlapping documented development consumption or unresolved usage. Never choose a range using new NLL, model predictions, improvement or cell exclusion. If no common clean 40-second interval exists, stop. Do not shorten the duration or substitute the old validation set to rescue the experiment.

The complete TEST_PROTOCOL.md will be written and hashed only after the data-usage gate passes, before frozen replay and new held-out model outputs. An unsuccessful data gate does not consume a confirmatory test and cannot receive a prediction-replication verdict.
