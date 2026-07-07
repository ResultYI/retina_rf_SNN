#!/usr/bin/env bash
set -euo pipefail

script_path="${BASH_SOURCE[0]}"
script_dir="${script_path%/*}"
if [[ "$script_dir" == "$script_path" ]]; then
  script_dir="."
fi
root_dir="$(cd "$script_dir/.." && pwd)"
matlab_dir="$root_dir/scripts/matlab"
if command -v cygpath >/dev/null 2>&1; then
  matlab_dir="$(cygpath -w "$matlab_dir")"
fi
matlab_dir="${matlab_dir//\'/\'\'}"
matlab_bin="${MATLAB_BIN:-matlab}"

"$matlab_bin" -batch "addpath('$matlab_dir'); check_isetbio_env"
