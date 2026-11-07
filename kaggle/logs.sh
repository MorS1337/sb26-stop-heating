#!/usr/bin/env bash
# Download a kernel's log and output files (submission.parquet on a real run).
#
#   kaggle/logs.sh sbx-pack-test [version]
#
# Output lands in kaggle/build/<slug>/output[/<version>] so it sits next to the
# notebook it came from. `kaggle kernels output` fetches the log only once the
# version has finished; check kaggle/status.sh first.
set -euo pipefail

slug="${1:?usage: kaggle/logs.sh <slug> [version]}"
version="${2:-}"
owner="${KAGGLE_USER:-mors1337}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$repo_root/kaggle/build/$slug/output"
if [[ -n "$version" ]]; then
  dest="$dest/$version"
  kernel="$owner/$slug/$version"
else
  kernel="$owner/$slug"
fi
mkdir -p "$dest"

exec kaggle kernels output "$kernel" -p "$dest" -o
