"""OracleNoopCtrlW -- CtrlW that receives the oracle payload and ignores it.

The non-perturbation probe for ``sbx.evalrun --oracle``: identical behaviour to
``CtrlW`` (the payload is accepted and dropped), so its totals must be
bit-identical to a normal run. Any difference means the channel - a reward-info
read, a weather query, the per-step hook call - moved the shared occupancy
``RandomState`` or the simulator clock.
"""

from __future__ import annotations

from .ctrl_w import CtrlW


class OracleNoopCtrlW(CtrlW):
    """CtrlW with an ``observe_oracle`` that deliberately does nothing."""

    name = "oracle_noop_ctrl_w"

    def observe_oracle(self, info):  # noqa: ARG002 - the payload is the point
        return None
