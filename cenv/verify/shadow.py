"""Shadow-verification harness entry point (``python -m cenv.verify.shadow``).

The harness itself is :mod:`cenv.shadow` -- component modules register their
Python-vs-C pairs through ``shadow_pairs(shadow, sim_root)`` and point at that
module by name, so the implementation lives there and this module is the
``cenv/verify/`` entry point plus the CLI.

    starter-kit/.venv/bin/python -m cenv.verify.shadow --component bookkeeping \
        --scenario building_train_8/fairbanks_8 --steps 200 \
        --policy policies.ctrl_w:CtrlW
"""

from __future__ import annotations

from cenv.shadow import (  # noqa: F401  (re-exported for peers' own scripts)
    ABS_TOL,
    REL_TOL,
    Metric,
    Shadow,
    main,
    run,
)

if __name__ == "__main__":
    raise SystemExit(main())
