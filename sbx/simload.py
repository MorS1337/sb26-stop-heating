"""Selects which working copy of the ``smart_buildings`` simulator a process imports.

The workspace keeps two full copies of the simulator package (``sims/shipped``
and ``sims/corrected``) so that a run can be pinned to a known source tree:

* ``shipped``   -- byte copy of starter-kit HEAD (95cea5b), i.e. exactly what the
                   public notebook / gateway runs.
* ``corrected`` -- the same tree with starter-kit PR #2 (energy-balanced hot-water
                   tank) and PR #3 (airflow-weighted return-air mixing) applied.

Why this module exists
----------------------
``pip install -e starter-kit`` installs an *editable* package as a
``sys.meta_path`` finder (``__editable___..._finder.py``). Meta-path finders are
consulted before ``sys.path``, so merely prepending ``sims/corrected`` to
``sys.path`` is silently ignored: ``import smart_buildings`` still lands in
``starter-kit/``. :func:`use_sim` therefore

1. purges any already-imported ``smart_buildings*`` modules,
2. removes the editable-install finders from ``sys.meta_path``,
3. puts the chosen tree root first on ``sys.path`` and makes it the working
   directory (every generated ``config.gin`` names its data files with
   ``smart_buildings/...`` paths relative to the tree root),
4. imports the package and asserts that ``smart_buildings.__file__`` really is
   inside the chosen tree.

Call :func:`use_sim` once, before anything imports ``smart_buildings``. One
process = one simulator tree: gin registers its configurables by module name, so
importing both copies in one process is not supported.

CLI check::

    python -m sbx.simload shipped
    python -m sbx.simload corrected
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]

SIM_ROOTS: dict[str, Path] = {
    "shipped": WORKSPACE / "sims" / "shipped",
    "corrected": WORKSPACE / "sims" / "corrected",
}
SIM_NAMES = tuple(SIM_ROOTS)


def sim_root(sim: str) -> Path:
    """Absolute path of a simulator tree, validated to look like one."""
    if sim not in SIM_ROOTS:
        raise ValueError(f"unknown sim {sim!r}; choose one of {SIM_NAMES}")
    root = SIM_ROOTS[sim].resolve()
    init = root / "smart_buildings" / "__init__.py"
    if not init.exists():
        raise FileNotFoundError(
            f"sim {sim!r} is not built: {init} is missing (run the harness copy step)"
        )
    return root


def _drop_editable_finders() -> list[str]:
    """Removes editable-install meta path finders; returns what was removed."""
    removed: list[str] = []
    for finder in list(sys.meta_path):
        owner = getattr(finder, "__module__", "") or ""
        marker = f"{owner} {type(finder).__name__}".lower()
        if "editable" in marker:
            sys.meta_path.remove(finder)
            removed.append(f"{owner}:{type(finder).__name__}")
    return removed


def use_sim(sim: str):
    """Imports ``smart_buildings`` from the ``sim`` tree and returns the module."""
    root = sim_root(sim)

    for name in [
        m for m in sys.modules if m == "smart_buildings" or m.startswith("smart_buildings.")
    ]:
        del sys.modules[name]
    _drop_editable_finders()

    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    # Keep the workspace importable (sbx/, policies/) after the chdir below.
    ws = str(WORKSPACE)
    if ws not in sys.path:
        sys.path.append(ws)

    os.chdir(root)  # config.gin data paths are relative to the tree root

    import smart_buildings  # noqa: PLC0415  (import order is the whole point)

    where = Path(smart_buildings.__file__).resolve()
    if root not in where.parents:
        raise RuntimeError(
            f"imported smart_buildings from {where}, expected a file under {root}; "
            "an editable install or a stale sys.path entry is shadowing the sim copy"
        )
    return smart_buildings


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sim", nargs="?", default="shipped", choices=SIM_NAMES)
    args = parser.parse_args(argv)
    mod = use_sim(args.sim)
    from smart_buildings import evaluate  # noqa: PLC0415

    print(f"sim            : {args.sim}")
    print(f"root (cwd)     : {os.getcwd()}")
    print(f"__file__       : {mod.__file__}")
    print(f"evaluate       : {evaluate.__file__}")
    print(f"scenarios root : {Path('smart_buildings/training_scenarios').resolve()}")


if __name__ == "__main__":
    main()
