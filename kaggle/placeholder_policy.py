"""Zero-action placeholder policy used to smoke-test the submission pipeline.

Deliberately identical in behaviour to the official starter notebook's
``TransferPolicy``: hold every continuous setpoint at the midpoint of its range
and switch every device on. It is the reference point the leaderboard floors
are measured against, so a notebook built from this file must reproduce the
public baseline score; anything else means the pipeline mangles the actions.

This module must stay importable with only numpy available: the builder embeds
it into a Kaggle notebook verbatim, and the notebook container cannot import
anything from this repository (or from ``smart_buildings``).
"""

import numpy as np


class ZeroActionPolicy:
    """Constant mid-range setpoints, all devices on."""

    # Present (though unused) so the builder's --param path is exercised end to end.
    PARAMS: dict = {"log_every": 0}

    def __init__(self, building_meta_data, action_spec):
        zone_count = len(building_meta_data["zone_temp_indices"])
        print(f"Controlling a building with {zone_count} zones", flush=True)

        continuous = action_spec["continuous_action"]
        discrete = action_spec["discrete_action"]
        # Clipped into the space's own bounds: a building whose setpoint ranges do
        # not straddle zero still gets a valid action, and the gateway rejects
        # anything out of range.
        self.action = {
            "continuous_action": np.clip(
                np.zeros(continuous.shape), continuous.low, continuous.high
            ).astype(np.float32),
            "discrete_action": np.clip(
                np.ones(discrete.shape), discrete.low, discrete.high
            ).astype(np.int32),
        }
        self._steps = 0

    def get_action(self, obs):
        self._steps += 1
        if self.PARAMS.get("log_every") and self._steps % int(self.PARAMS["log_every"]) == 0:
            print(f"step {self._steps}: holding {len(self.action['continuous_action'])} setpoints", flush=True)
        return self.action
