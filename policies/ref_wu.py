"""Reference policy: danielwuu's "CPU A — alpha 0.75 control" controller.

Notebook: https://www.kaggle.com/code/danielwuu/cpu-a-alpha075-control-20261022
Author: Daniel Wu. The controller itself is not in the notebook source: the
notebook embeds a SHA-256-verified base64 ZIP with

    policy.py          28,149 B  the analytic controller (per-zone RLS model,
                                 1-step delay compensation, joint per-AHU
                                 (supply temperature, recirculation) search)
    round2/runtime.py   7,744 B  the two-MLP residual-drift ensemble and the
                                 bounded-correction wrapper
    member_0.npz / member_1.npz  2 x 266 KB, the trained MLPs

That archive is decoded verbatim (archive sha256
8819406ae7eed24385d6560efb72586af19a77943a900d4f7db412fd6013e9ff) into
`refs/notebooks/danielwuu_cpu-a-alpha075-control-20261022/decoded/`, and this
module is only the adapter the notebook itself uses:

    ResidualPolicy(building_meta_data, action_spec,
                   ensemble=ResidualEnsemble([member_0.npz, member_1.npz]),
                   alpha=0.75, correction_limit=0.25, disagreement_limit=0.15)

plus the notebook's dtype/shape/bounds check on the returned action.

This is a LOCAL reference only (it reads the decoded files from an absolute
path); it is never submitted.

Notebook-logged reference numbers (Kaggle gateway run, `building_train_1/
atlanta_2`, 8,640 steps): mean reward -0.00340; public LB 0.01062.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

DECODED = Path(
    "/Users/mors1337/Code/DL/doma/refs/notebooks/"
    "danielwuu_cpu-a-alpha075-control-20261022/decoded"
)
MEMBERS = ("member_0.npz", "member_1.npz")
# The notebook's own CORE_SHA256, re-checked here so a stale or hand-edited
# decode cannot silently change what we benchmark.
CORE_SHA256 = {
    "member_0.npz": "3a7338defdb6407e5050cf0226a43fb70f44f4e058459ea9f0cb3dc1088ac577",
    "member_1.npz": "7cb7aa9d41f4681054b59eb2043e4147a5ce359cd4d051b6b43f1b3ab3aefddc",
    "policy.py": "185eccbe60fdfc3728b03ecf834de9a2e58943d241d25ce16a1012a043bb33f6",
    "round2/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "round2/runtime.py": "00dab1428d296ac65cb8b9bbb16092a3c49482fc81cd2644f2fe17ac257b28c3",
}
GUARDS = {"alpha": 0.75, "correction_limit": 0.25, "disagreement_limit": 0.15}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_vendor():
    """Import the decoded `policy.py` and `round2/runtime.py`.

    `round2/runtime.py` does `from policy import TransferPolicy`, so the decoded
    `policy.py` has to be importable under that exact top-level name.
    """
    for name, expected in CORE_SHA256.items():
        actual = _sha256(DECODED / name)
        if actual != expected:
            raise ValueError(f"{name} does not match the notebook's checksum")
    if "policy" in sys.modules and getattr(sys.modules["policy"], "__file__", None) != str(
        DECODED / "policy.py"
    ):
        raise RuntimeError("a different module named 'policy' is already imported")
    core = sys.modules.get("policy") or _load_module("policy", DECODED / "policy.py")
    runtime = _load_module("_wu_round2_runtime", DECODED / "round2" / "runtime.py")
    return core, runtime


_CORE, _RUNTIME = _load_vendor()


class RefWu:
    """The notebook's `TransferPolicy` adapter around `ResidualPolicy`."""

    def __init__(self, building_meta_data, action_spec):
        self._spec = action_spec
        ensemble = _RUNTIME.ResidualEnsemble([DECODED / name for name in MEMBERS])
        self._policy = _RUNTIME.ResidualPolicy(
            building_meta_data, action_spec, ensemble=ensemble, **GUARDS
        )

    def get_action(self, obs):
        action = self._policy.get_action(obs)  # exactly one stateful call
        checked = {}
        for key, dtype in (("continuous_action", np.float32),
                           ("discrete_action", np.int32)):
            values, space = np.asarray(action[key]), self._spec[key]
            if values.shape != tuple(space.shape):
                raise ValueError("Invalid action shape: " + key)
            if not np.isfinite(values).all():
                raise ValueError("Non-finite policy action: " + key)
            if key == "discrete_action" and not np.isin(values, (0, 1)).all():
                raise ValueError("Discrete actions must be 0 or 1")
            checked[key] = np.clip(values, space.low, space.high).astype(dtype)
        return checked
