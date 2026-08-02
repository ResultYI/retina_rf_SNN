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
    TeacherPopulationConfig,
    generate_teacher_responses,
)
from data.cone_response import ConeResponseExport
from data.dataset import (
    validate_compatible_cone_exports,
    validate_loaded_cone_exports,
)
from data.input_identity import DatasetKind
from data.rgc_response_export import write_rgc_response
from data.synthetic_teacher import (
    TeacherInputNormalization,
    fit_teacher_input_normalization,
)


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
    parser.add_argument("--cells-per-type-polarity", type=int, default=4)
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
    all_paths = (*train_paths, *validation_paths, *test_paths)
    all_exports = _load_exports(all_paths)
    train_end = len(train_paths)
    validation_end = train_end + len(validation_paths)
    train_exports = all_exports[:train_end]
    teacher_normalization = fit_teacher_input_normalization(
        np.stack([export.response for export in train_exports])
    )
    validation_exports = all_exports[train_end:validation_end]
    test_exports = all_exports[validation_end:]
    population_config = TeacherPopulationConfig(args.cells_per_type_polarity)
    output = Path(args.output_dir)
    for split, exports, paths, offset in (
        ("train", train_exports, train_paths, 0),
        ("validation", validation_exports, validation_paths, 1000),
        ("test", test_exports, test_paths, 2000),
    ):
        result = _generate_split(
            exports,
            paths,
            trials=args.trials,
            seed=args.seed + offset,
            adaptive=args.teacher == "adaptive",
            teacher_normalization=teacher_normalization,
            population_config=population_config,
            teacher_seed=args.seed,
        )
        write_rgc_response(
            output / f"{split}.h5",
            result.session,
            teacher_kernels=result.kernels,
            teacher_normalization=result.teacher_normalization,
        )


def _generate_split(
    exports: tuple[ConeResponseExport, ...],
    paths: tuple[Path, ...],
    *,
    trials: int,
    seed: int,
    adaptive: bool,
    teacher_normalization: TeacherInputNormalization,
    population_config: TeacherPopulationConfig,
    teacher_seed: int,
) -> SyntheticTeacherResult:
    validate_loaded_cone_exports(exports)
    if any(
        export.input_identity.dataset_kind
        is not DatasetKind.SYNTHETIC_METHOD_VALIDATION
        for export in exports
    ):
        raise SyntheticBenchmarkError(
            "Synthetic benchmark inputs must declare synthetic_method_validation"
        )
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
        teacher_normalization=teacher_normalization,
        population_config=population_config,
        teacher_seed=teacher_seed,
        input_identity=exports[0].input_identity.with_sources(
            tuple(
                export.input_identity.stimulus_source_fingerprints[0]
                for export in exports
            )
        ),
    )


def _load_exports(paths: tuple[Path, ...]) -> tuple[ConeResponseExport, ...]:
    return validate_compatible_cone_exports(paths)


if __name__ == "__main__":
    main()
