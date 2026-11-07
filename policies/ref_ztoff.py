"""Reference policy: the `zt_off` base rule of ndaedzomukhuba's "sb-alex-rl".

Notebook: https://www.kaggle.com/code/ndaedzomukhuba/sb-alex-rl
Author: alexchilton (Kaggle handle ndaedzomukhuba). Its rule `zt_off` is
malyshevdanil's public per-zone PI controller, https://www.kaggle.com/code/malyshevdanil/smart-buildings-probe-zt-best
(Apache-2.0), in its PI hours, plus an off-hours rule: outside those hours every
continuous command is at its minimum (-1), the recirculation dampers are held at
their maximum (1.0) and every on/off switch is off.

This module ports that rule WITHOUT the PPO residual on top of it, i.e. the
`skynet/base.py::ZTOffFloor.propose` behaviour with a zero correction:

    pc, pd = zt.get_action(obs)          # the rule always runs, so its state
    free = bool(zt._pi_active)           # (integrals, holds) stays current
    return (pc, pd) if free else (off_c, off_d)

where `off_c = -1` everywhere except `recirculation_indices` (= +1) and
`off_d = 0` everywhere. See `refs/notebooks/ndaedzomukhuba_sb-alex-rl/` and the
`sb/skynet/base.py` copy under `scratch/NotebookIntel/out_ndaedzomukhuba_sb-alex-rl/`.

The zt-best part is imported from our own `policies/ref_ztbest.py` (a faithful
port of the notebook, including the vendored Apache-2.0 organizer PID chain), so
the two modules cannot drift apart.

Notebook-logged reference numbers: the rule alone scores 0.01251 on held-out
building 8 (the full 30-scenario leaderboard of the RL-on-top version is
0.01128); the zt-best part alone scores -0.00556 mean reward on
`building_train_1/atlanta_2` and 0.01420 on the public LB.
"""

import numpy as np

from policies.ref_ztbest import RefZtBest


class RefZtOff(RefZtBest):
    """zt-best inside its PI hours, everything off outside them."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self.off_continuous = np.full(self.n_continuous, -1.0, dtype=np.float32)
        self.off_continuous[list(self.recirculation_indices)] = 1.0
        self.off_discrete = np.zeros(self.n_discrete, dtype=np.int32)

    def get_action(self, obs):
        # Run the rule unconditionally: its PI state and cooling holds have to
        # stay current so the first PI hour of the day starts from where the
        # rule left them.
        action = super().get_action(obs)
        if self._pi_active:
            return action
        return {
            "continuous_action": self.off_continuous.copy(),
            "discrete_action": self.off_discrete.copy(),
        }
