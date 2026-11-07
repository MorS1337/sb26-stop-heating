"""Fidelity plots from the error series fastsim.replay --err-dir writes.

Panel per building: the max over zones, and the mean over zones, of the
zone-mean temperature error against the real rollout, as a function of horizon,
one line per basis size. Both are on a log time axis so the first hour is
visible next to the ten days.

Usage:
  starter-kit/.venv/bin/python -m fastsim.plot --err-dir fastsim/plots/err \
      --out fastsim/plots/fidelity.png
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--err-dir", default="fastsim/plots/err")
    p.add_argument("--out", default="fastsim/plots/fidelity.png")
    a = p.parse_args(argv)
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    files = sorted(glob.glob(str(Path(a.err_dir) / "*.npz")))
    if not files:
        raise FileNotFoundError(f"no error series in {a.err_dir}")
    by_building: dict[str, list] = {}
    for f in files:
        m = re.match(r"(.+?)__(.+?)__m(\d+)\.npz", Path(f).name)
        building, kind, mval = m.group(1), m.group(2), int(m.group(3))
        by_building.setdefault(building, []).append((mval, np.load(f)))
    buildings = sorted(by_building)
    fig, axes = plt.subplots(1, len(buildings), figsize=(5.5 * len(buildings), 4.0),
                             squeeze=False)
    for ax, b in zip(axes[0], buildings):
        for mval, d in sorted(by_building[b]):
            err = d["err"][0]
            t = d["t_hours"] / 24.0
            ax.semilogy(t[1:], np.abs(err[1:]).max(axis=1), lw=1.2,
                        label=f"m={mval} max")
            ax.semilogy(t[1:], np.abs(err[1:]).mean(axis=1), lw=1.0, ls="--",
                        label=f"m={mval} mean")
        ax.axhline(0.1, color="k", lw=0.8, ls=":")
        ax.set_xlabel("simulated day")
        ax.set_ylabel("zone-mean error vs real rollout (K)")
        ax.set_title(b)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
