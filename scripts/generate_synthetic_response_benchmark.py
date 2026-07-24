from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.point_process_teacher import (
    SyntheticTeacherResult,
    generate_teacher_responses,
)
from data.cone_response import load_cone_response
from data.rgc_response_export import write_rgc_response


class SyntheticBenchmarkError(ValueError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic recorded-RGC response benchmark."
    )
    parser.add_argument("--train-glob", required=True)
    parser.add_argument("--validation-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher", choices=("static", "adaptive"), default="adaptive")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--test-count", type=int, default=2)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    train_paths = tuple(Path(path) for path in sorted(glob.glob(args.train_glob)))
    validation_paths = tuple(
        Path(path) for path in sorted(glob.glob(args.validation_glob))
    )
    if args.train_limit is not None:
        train_paths = train_paths[: args.train_limit + args.test_count]
    if args.validation_limit is not None:
        validation_paths = validation_paths[: args.validation_limit]
    if len(train_paths) <= args.test_count or not validation_paths:
        raise SyntheticBenchmarkError(
            "Synthetic benchmark needs train, validation, and held-out files"
        )
    test_paths = train_paths[-args.test_count :]
    train_paths = train_paths[: -args.test_count]
    output = Path(args.output_dir)
    for split, paths, offset in (
        ("train", train_paths, 0),
        ("validation", validation_paths, 1000),
        ("test", test_paths, 2000),
    ):
        result = _generate_split(
            paths,
            trials=args.trials,
            seed=args.seed + offset,
            adaptive=args.teacher == "adaptive",
        )
        write_rgc_response(
            output / f"{split}.h5",
            result.session,
            teacher_kernels=result.kernels,
        )


def _generate_split(
    paths: tuple[Path, ...],
    *,
    trials: int,
    seed: int,
    adaptive: bool,
) -> SyntheticTeacherResult:
    exports = tuple(load_cone_response(path) for path in paths)
    shape = exports[0].response.shape
    if any(export.response.shape != shape for export in exports):
        raise SyntheticBenchmarkError("Synthetic source cone shapes must match")
    return generate_teacher_responses(
        np.stack([export.response for export in exports]),
        exports[0].positions_degs,
        tuple(export.source_id or path.stem for export, path in zip(exports, paths)),
        exports[0].time_axis_seconds,
        trials=trials,
        seed=seed,
        adaptive=adaptive,
    )


if __name__ == "__main__":
    main()
