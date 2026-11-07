#!/usr/bin/env bash
# Report a kernel's status, e.g. "KernelWorkerStatus.COMPLETE".
#
#   kaggle/status.sh sbx-ctrl-v1
#
# Kaggle-side execution is asynchronous: push returns immediately and the
# notebook runs in the background. Poll until COMPLETE (or ERROR), then read the
# log with kaggle/logs.sh <slug>.
set -euo pipefail

slug="${1:?usage: kaggle/status.sh <slug>}"
owner="${KAGGLE_USER:-mors1337}"
exec kaggle kernels status "$owner/$slug"
