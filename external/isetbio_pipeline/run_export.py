from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
MATLAB_DIR = Path(__file__).resolve().parent / "matlab"
DEFAULT_INPUT = ROOT / "data" / "isetbio" / "input_movie.mp4"
DEFAULT_OUTPUT = ROOT / "results" / "isetbio" / "input_cone_response_movie.h5"


def matlab_quote(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def resolve_matlab() -> str:
    candidate = shutil.which("matlab")
    if candidate:
        return candidate
    standard = Path(r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe")
    if standard.is_file():
        return str(standard)
    raise FileNotFoundError("MATLAB executable was not found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Video file, frame directory, or a still image for compatibility.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--time-steps",
        type=int,
        default=8,
        help="Max frames for video/frame directories; repeat count for still images.",
    )
    parser.add_argument(
        "--eye-movements",
        action="store_true",
        help="Apply fixational drift (requires MATLAB Statistics Toolbox).",
    )
    args = parser.parse_args()

    if args.time_steps <= 0:
        parser.error("--time-steps must be positive")
    if not (args.input.is_file() or args.input.is_dir()):
        parser.error(f"input video, frame directory, or image does not exist: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matlab_expression = (
        f"addpath('{matlab_quote(MATLAB_DIR)}'); "
        f"export_cone_response('{matlab_quote(args.input)}', "
        f"'{matlab_quote(args.output)}', {args.time_steps}, "
        f"{str(args.eye_movements).lower()})"
    )
    command = [resolve_matlab(), "-batch", matlab_expression]
    print("Running:", " ".join(command[:2]), "<MATLAB expression>")
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
