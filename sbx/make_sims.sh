#!/usr/bin/env bash
# Recreate the two simulator trees from the pristine starter-kit clone.
#
#   sbx/make_sims.sh            # skips trees that already exist
#   sbx/make_sims.sh --force    # rebuilds both (deletes generated pools inside them)
#
# shipped   = starter-kit at commit 95cea5b (tracked files only, via git archive)
# corrected = shipped + starter-kit PR #2 (energy-balanced hot-water tank)
#                     + PR #3 (airflow-weighted return-air mixing),
#             patches frozen in sbx/patches/ so no GitHub access is needed.
#
# Afterwards populate both trees:
#   starter-kit/.venv/bin/python -m sbx.make_pool
#   starter-kit/.venv/bin/python sbx/weather_ext.py --copy-to sims/shipped/smart_buildings/weather
#   starter-kit/.venv/bin/python sbx/weather_ext.py --copy-to sims/corrected/smart_buildings/weather
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kit="$root/starter-kit"
commit="95cea5b"
force=0
[[ "${1:-}" == "--force" ]] && force=1

if [[ ! -d "$kit/.git" ]]; then
  echo "missing $kit: git clone https://github.com/smart-buildings-2026/starter-kit.git" >&2
  exit 1
fi

for sim in shipped corrected; do
  dest="$root/sims/$sim"
  if [[ -e "$dest" && $force -eq 0 ]]; then
    echo "exists, skipped: $dest (pass --force to rebuild)"
    continue
  fi
  rm -rf "$dest"
  mkdir -p "$dest"
  git -C "$kit" archive "$commit" | tar -x -C "$dest"
  if [[ "$sim" == "corrected" ]]; then
    for patch in "$root"/sbx/patches/pr2_*.diff "$root"/sbx/patches/pr3_*.diff; do
      (cd "$dest" && git apply --whitespace=nowarn "$patch")
      echo "applied $(basename "$patch") -> $dest"
    done
  fi
  echo "built $dest"
done
