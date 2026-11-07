#!/usr/bin/env bash
# Submit a finished kernel version to the code competition.
#
#   kaggle/submit.sh sbx-ctrl-v1 3 "ctrl-v1 occupancy-aware setpoints"
#
# For a code competition the CLI takes the kernel owner/slug, the version number
# and the *name of the output file the kernel produces* (the gateway writes
# submission.parquet itself, so there is nothing to upload). Verified against
# `kaggle competitions submit -h` (CLI 2.2.1):
#
#   -k, --kernel KERNEL    Name of kernel (notebook) to submit to a code competition
#   -v, --version VERSION  Version of kernel to submit to a code competition
#   -f, --file FILE_NAME   Name of the output file produced by a kernel
#
# The competition is pinned to its slug so a stale `kaggle config set` cannot
# redirect the submission. Reserved for real controllers: a submission consumes
# the day's allowance and shows up on the leaderboard.
set -euo pipefail

slug="${1:?usage: kaggle/submit.sh <slug> <version> <message>}"
version="${2:?usage: kaggle/submit.sh <slug> <version> <message>}"
message="${3:?usage: kaggle/submit.sh <slug> <version> <message>}"
owner="${KAGGLE_USER:-mors1337}"
competition="neurips-2026-smart-buildings"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
notebook="$repo_root/kaggle/build/$slug/submission.ipynb"

# A --smoke-steps build truncates every episode and must never reach the
# leaderboard. The local copy is the best evidence of what the pushed version
# contains, so refuse unless the caller explicitly overrides.
if [[ -f "$notebook" ]] && grep -q 'TEST BUILD' "$notebook"; then
  if [[ "${SB_ALLOW_TEST_SUBMIT:-0}" != "1" ]]; then
    echo "$notebook is a TEST BUILD (episodes truncated); rebuild without --smoke-steps." >&2
    echo "set SB_ALLOW_TEST_SUBMIT=1 only if you are certain this is intended." >&2
    exit 1
  fi
  echo "warning: submitting a TEST BUILD (SB_ALLOW_TEST_SUBMIT=1)" >&2
fi

exec kaggle competitions submit "$competition" \
  -k "$owner/$slug" \
  -v "$version" \
  -f submission.parquet \
  -m "$message"
