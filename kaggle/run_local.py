#!/usr/bin/env python3
"""Execute a generated submission notebook against the local gateway.

This is the pre-flight for ``kaggle/push.sh``: it runs the notebook's code cells
exactly as written (same imports, same embedded policy modules, same inference
server call) but with ``competition_data_folder`` pointed at a local copy of the
competition data and episodes truncated to a few steps.

Usage:
    starter-kit/.venv/bin/python kaggle/run_local.py \
        kaggle/build/sbx-pack-test/submission.ipynb --max-steps 30

Only the public evaluation scenario (``building_train_1/atlanta_2``) is mounted
in the public data copy, so a green run here proves the notebook and the policy
are wired correctly -- not that they scale to the 30 hidden scenarios.

What it changes relative to Kaggle, and nothing else:

* ``SB_COMPETITION_DATA_FOLDER`` points the notebook at ``comp_data/``;
* ``HEARTH_HOST_SMART_BUILDINGS_MAX_STEPS`` truncates every episode (the gateway
  refuses to run untruncated without a rerun flag);
* the editable install of ``smart_buildings`` from ``starter-kit`` is removed from
  the import path, so the simulator that loads is the one on the data mount --
  exactly what happens inside the submission container;
* numba/matplotlib caches go to a temp dir, because the data copy is read-only.

polars is missing from the workspace venv (the gateway needs it only for the
submission table), and so is ``grpcio-tools`` (the relay compiles its protobuf
stubs on first import; the Kaggle image ships it). Put either in
``kaggle/.pydeps`` -- the script loads that directory automatically -- or point
at another site-packages with ``--extra-path``:

    uv pip install --target kaggle/.pydeps --python starter-kit/.venv/bin/python \\
        grpcio-tools polars
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPETITION_DATA = REPO_ROOT / "comp_data"
DEFAULT_MAX_STEPS = 30
# Directories searched for polars when it is not importable in the active venv.
_POLARS_FALLBACK_GLOBS = (
    "/opt/homebrew/Caskroom/miniconda/base/lib/python3*/site-packages",
    "~/miniconda3/lib/python3*/site-packages",
    "~/anaconda3/lib/python3*/site-packages",
    "/opt/conda/lib/python3*/site-packages",
)


def _drop_editable_finder() -> None:
    """Remove the setuptools editable-install finder from ``sys.meta_path``.

    ``smart_buildings`` is editable-installed from ``starter-kit``; the finder is a
    meta-path hook, so it wins over ``sys.path`` and would silently load a
    different simulator copy than the one the gateway mounts.
    """
    kept = []
    for finder in sys.meta_path:
        module = getattr(finder, "__module__", "") or type(finder).__module__
        cls = type(finder).__name__
        if module.startswith("__editable__") or "editable" in cls.lower():
            continue
        kept.append(finder)
    sys.meta_path[:] = kept
    for name in [name for name in sys.modules if name == "smart_buildings" or name.startswith("smart_buildings.")]:
        del sys.modules[name]


def _ensure_polars(extra_paths: list[str]) -> None:
    try:
        import polars  # noqa: F401
        return
    except ImportError:
        pass
    candidates = list(extra_paths)
    for pattern in _POLARS_FALLBACK_GLOBS:
        candidates.extend(sorted(glob.glob(os.path.expanduser(pattern))))
    for candidate in candidates:
        if not Path(candidate, "polars").is_dir():
            continue
        # Appended, not inserted: the workspace venv's numpy/scipy must keep winning.
        sys.path.append(candidate)
        try:
            import polars  # noqa: F401
        except ImportError:
            sys.path.remove(candidate)
            continue
        print(f"[local] polars from {candidate}", flush=True)
        return
    raise SystemExit(
        "polars is not importable; install it or pass --extra-path <site-packages containing polars>"
    )


def _code_cells(notebook_path: Path) -> list[str]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebook", type=Path, help="generated submission.ipynb")
    parser.add_argument(
        "--competition-data",
        type=Path,
        default=DEFAULT_COMPETITION_DATA,
        help=f"folder containing smart_buildings/ and kaggle_evaluation/ (default: {DEFAULT_COMPETITION_DATA})",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="truncate every episode")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="directory to run in; submission.parquet lands here (default: a fresh temp dir)",
    )
    parser.add_argument(
        "--extra-path",
        action="append",
        default=[],
        help="extra directory to search for missing runtime packages (polars)",
    )
    args = parser.parse_args(argv)

    notebook_path = args.notebook.resolve()
    competition_data = args.competition_data.resolve()
    if not (competition_data / "smart_buildings").is_dir():
        raise SystemExit(f"{competition_data} does not contain smart_buildings/")
    if not (competition_data / "smart_buildings" / "evaluation_scenarios").is_dir():
        raise SystemExit(f"{competition_data} has no evaluation_scenarios/ to run against")

    # Kaggle's own cache redirect targets /tmp; mirror it so nothing is written into
    # the read-only data copy.
    scratch = Path(tempfile.mkdtemp(prefix="sbx_local_"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(scratch / "numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(scratch / "matplotlib"))
    for path in (os.environ["NUMBA_CACHE_DIR"], os.environ["MPLCONFIGDIR"]):
        Path(path).mkdir(parents=True, exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"
    # A leftover rerun flag would make the server wait for a host that is not there.
    os.environ.pop("KAGGLE_IS_COMPETITION_RERUN", None)
    os.environ["SB_COMPETITION_DATA_FOLDER"] = str(competition_data) + "/"
    os.environ["SB_MODULE_DIR"] = str(scratch / "policy_modules")
    os.environ["HEARTH_HOST_SMART_BUILDINGS_MAX_STEPS"] = str(args.max_steps)
    # The policy may print per-step diagnostics; keep our own logging ordered.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    _drop_editable_finder()
    # Extra site-packages come first: the relay's protobuf stubs must be compiled
    # against the protobuf that ships with grpcio-tools, not the older one in the
    # venv. The notebook later inserts the competition folders ahead of these.
    for directory in [REPO_ROOT / "kaggle" / ".pydeps", *(Path(item) for item in args.extra_path)]:
        if directory.is_dir():
            sys.path.insert(0, str(directory.resolve()))
    _ensure_polars(args.extra_path)

    cells = _code_cells(notebook_path)
    if not cells:
        raise SystemExit(f"{notebook_path} has no code cells")

    # The gateway writes submission.parquet into the process working directory.
    # Run from a scratch directory so a smoke-test file never looks like a real one.
    work_dir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="sbx_local_run_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)

    print(f"[local] executing {len(cells)} code cells from {notebook_path.name}", flush=True)
    print(f"[local] data={competition_data} max_steps={args.max_steps} work_dir={work_dir}", flush=True)

    namespace: dict = {"__name__": "__main__", "__file__": str(notebook_path)}
    try:
        for index, source in enumerate(cells, start=1):
            print(f"[local] --- cell {index} ---", flush=True)
            exec(compile(source, f"{notebook_path.name}:cell{index}", "exec"), namespace)
    except Exception:
        traceback.print_exc()
        print("[local] FAILED", flush=True)
        return 1

    print("[local] notebook completed; the gateway log above carries the score", flush=True)
    produced = sorted(work_dir.glob("*.parquet"))
    if produced:
        print(f"[local] submission file: {produced[0]}", flush=True)
    else:
        print("[local] WARNING: no submission parquet was written", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
