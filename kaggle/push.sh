#!/usr/bin/env bash
# Push one built submission notebook to Kaggle.
#
#   kaggle/push.sh sbx-ctrl-v1
#
# The kernel id, title and competition sources come from
# kaggle/build/<slug>/kernel-metadata.json (private, CPU, internet off). The
# push publishes a new version; follow it with kaggle/status.sh <slug>.
set -euo pipefail

slug="${1:?usage: kaggle/push.sh <slug>}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_root/kaggle/build/$slug"

if [[ ! -f "$build_dir/kernel-metadata.json" || ! -f "$build_dir/submission.ipynb" ]]; then
  echo "no build at $build_dir -- run: python kaggle/build_notebook.py --policy <mod.py:Class> --slug $slug" >&2
  exit 1
fi

# The account's GPU/TPU quota is reserved for other competitions: CPU kernels only.
if grep -Eq '"enable_(gpu|tpu)": *true' "$build_dir/kernel-metadata.json"; then
  echo "refusing to push $slug: kernel-metadata.json enables a GPU/TPU (CPU only)" >&2
  exit 1
fi

# package the kernel from its own directory: the CLI uploads code_file plus any
# sibling files it references, and rejects stray metadata elsewhere.
cd "$build_dir"
exec kaggle kernels push -p "$build_dir"
