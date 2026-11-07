"""CtrlMPC -- receding-horizon planner on the 79-node twin, OFFLINE EXPERIMENT.

This is not submission-safe (it imports scipy/HiGHS and loads a precomputed
model of one building's geometry) and is meant for offline capability
measurement only.  Documented here because the mapping matters:

  * every `solve_every` steps (1 h) the policy rebuilds the twin's state from
    the observed zone temperatures (each room's core and boundary sub-node set
    to its zone mean, the partition node and the interior mass to the mean of
    the rooms) and solves a linear programme over the next `lp_steps` 15-minute
    slots;
  * decisions per 15-minute slot and per zone: `qp - qm` = net heat delivered to
    the zone air (reheat when positive, extra cooling above the floor when
    negative) plus the extra airflow `f` needed to carry it.  Constraints: the
    twin's linear dynamics with the ventilation floor held at the legal minimum,
    the coil capacity `q <= (f_floor+f)*rho*cp*(T_discharge_max - T_sup)`, the
    reheat ventilation penalty `(1-r)*f*rho*cp*(T_mix - T_out)` priced as gas,
    and the *schedule* comfort band as a hard constraint for every zone during
    comfort mode -- no occupancy information at all;
  * objective: the reward's own hourly fuel/carbon weights, taken from the same
    relative tables CtrlW embeds (`ELEC_REL`, `GAS_RATIO`), which are
    proportional to the reward's marginal loss per watt;
  * the outside temperature is held at its last observed value for the whole
    horizon (no forecast), prices come from the clock, and the plan's zone
    temperatures for the next hour become the *band targets* that CtrlW's
    existing AHU search and VAV layer track.  So the plan acts through CtrlW's
    actuator mapping, not around it.

Everything CtrlW does not override (residual ensemble, fuel weights, release,
preheat ramp) stays as CtrlW has it; this class only replaces the band.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

try:  # packaged notebook: embedded modules are top-level names
    from ctrl_w import CtrlW
except ImportError:  # workspace: policies/ is a package
    from .ctrl_w import CtrlW

RHO_CP = 1.2 * 1006.0
DEFAULT_MODEL = ('/Users/mors1337/Code/DL/doma/scratch/OptGap/models/'
                 'building_train_1__fairbanks_1__shipped__sub2.npz')


class CtrlMPC(CtrlW):
    """CtrlW whose comfort band is set by a receding-horizon LP on the twin."""

    name = "ctrl_mpc"

    PARAMS = dict(CtrlW.PARAMS)
    PARAMS.update({
        "mpc_enabled": True,
        "model_path": DEFAULT_MODEL,
        "lp_steps": 32,          # 8 h at 15 min
        "lp_dt": 900.0,
        "solve_every": 12,       # re-solve once an hour (5-min sim steps)
        "plan_within_hour": 4,   # LP slots consumed per hour
        "target_width": 0.30,    # band half-width handed to CtrlW
        "follow_mode": "night_only",  # or "full"
        "night_floor": 280.0,    # never plan the night below this
        "alpha": None,           # envelope scale; None -> fit file if present
        "beta": None,
        "cool_gap": 12.0,
        "flow_cap": 3.0,
        # --- diagnostics that reproduce version 1 exactly by default --------
        "tcold_mode": "sup",          # 'zone' = physical cold side of the coil
        "heat_price_mode": "one",      # 'phys' = price the OA share at the cap
        "plan_margin": 0.0,           # K above the planned comfort floor
        "night_cap": 1.0e9,           # K ceiling on the plan's night floor
    })

    def __init__(self, building_meta_data, action_spec=None):
        super().__init__(building_meta_data, action_spec)
        p = dict(self.PARAMS)
        self.mpc_enabled = bool(p["mpc_enabled"])
        self._plan = None
        self._plan_at = -10 ** 9
        self._plan_targets = None
        self._lp_calls = 0
        self._lp_fail = 0
        self.model = None
        if not self.mpc_enabled:
            return
        try:
            self.model = _load_model(p["model_path"], float(p["lp_dt"]),
                                     p["alpha"], p["beta"])
        except Exception:  # noqa: BLE001 - offline experiment, fall back to CtrlW
            self.model = None
        self.lp_steps = int(p["lp_steps"])
        self.solve_every = max(1, int(p["solve_every"]))
        self.plan_within = max(1, int(p["plan_within_hour"]))
        self.width = float(p["target_width"])
        self.follow_full = str(p["follow_mode"]) == "full"
        self.night_floor = float(p["night_floor"])
        self.cool_gap = float(p["cool_gap"])
        self.flow_cap = float(p["flow_cap"])
        self.tcold_mode = str(p.get("tcold_mode", "sup"))
        self.heat_price_mode = str(p.get("heat_price_mode", "one"))
        self.plan_margin = float(p.get("plan_margin", 0.0))
        self.night_cap = float(p.get("night_cap", 1.0e9))

    # ------------------------------------------------------------------ band
    def _band(self, obs, temperature):
        low, high, now = super()._band(obs, temperature)
        if not self.mpc_enabled or self.model is None:
            return low, high, now
        try:
            self._maybe_solve(obs, np.asarray(temperature, float), now)
            target = self._target_for_step()
        except Exception:  # noqa: BLE001
            self._plan_targets = None
            return low, high, now
        if target is None:
            return low, high, now
        low_s = np.asarray(low, float)
        high_s = np.asarray(high, float)
        if now and not self.follow_full:
            # inside the comfort window keep CtrlW's own band (its margin is a
            # buffer against the disturbances the LP cannot see); the plan is
            # only allowed to ask for a *lower* floor
            lo = np.minimum(low_s, np.maximum(target, 0.0))
            hi = high_s
            return lo, hi, now
        lo = np.clip(target, low_s, high_s)
        if not now:
            # the plan sets the night *floor* only; tightening the upper edge
            # out of comfort mode makes the AHU search buy cooling it cannot
            # pay for in a cold climate (measured: -127.9 with a clamped high)
            lo = np.maximum(lo, min(self.night_floor, float(np.min(high_s))))
            if self.night_cap < 1.0e8:
                lo = np.minimum(lo, np.maximum(self.night_cap, low_s))
            return lo, high_s, now
        hi = np.minimum(high_s, lo + self.width)
        return lo, hi, now

    # ------------------------------------------------------------------- LP
    def _maybe_solve(self, obs, temperature, now):
        if self._plan_targets is not None and self.step_count - self._plan_at < self.plan_within:
            return
        if self._plan_targets is not None and self.step_count - self._plan_at < self.solve_every:
            return
        m = self.model
        Nr = m['N_room']
        if temperature.size != Nr:
            return
        outside = float(obs[self.outside_obs[0]]) if self.outside_obs else 290.0
        hour = float(self._hour(obs))
        weekday = self._weekday(obs)
        plan = _solve(m, temperature, outside, hour, weekday,
                      float(self.comfort_band[0]), float(self.comfort_band[1]),
                      float(self.eco_band[0]), float(self.eco_band[1]),
                      self.lp_steps, self.night_floor, self.cool_gap, self.flow_cap,
                      float(self.start_hour), self.tcold_mode, self.heat_price_mode,
                      self.plan_margin)
        self._lp_calls += 1
        if plan is None:
            self._lp_fail += 1
            return
        self._plan_targets = plan
        self._plan_at = self.step_count

    def _target_for_step(self):
        if self._plan_targets is None:
            return None
        k = self.step_count - self._plan_at
        if k < 0 or k >= len(self._plan_targets):
            return None
        # one LP slot covers lp_dt seconds = lp_dt/300 sim steps
        slot = k // max(1, int(round(self.model['dt'] / 300.0)))
        slot = min(slot, len(self._plan_targets) - 1)
        return self._plan_targets[slot]


# --------------------------------------------------------------------------- #
# twin model
# --------------------------------------------------------------------------- #
def _load_model(path, dt, alpha, beta):
    z = np.load(path, allow_pickle=True)
    m = {k: z[k] for k in z.files}
    nn = int(m['n_node'])
    Nr = int(m['N_room'])
    C300 = np.asarray(m['C'], float)          # heat capacity rate at 300 s
    r300 = np.asarray(m['r'], float)
    g300 = np.asarray(m['g'], float)
    Kdir = np.asarray(m['Kdir'], float)
    Kamb = np.asarray(m['Kamb'], float)
    Mz = np.asarray(m['M'], float)
    Koff = np.asarray(m['Koff'], float)
    Kdiag = np.asarray(m['Kdiag'], float)
    if alpha is None:
        try:
            gfile = np.load(path.replace('__sub2.npz', '__grey.npz'))
            alpha = float(gfile['alpha'])
            beta = float(gfile['beta'])
        except Exception:  # noqa: BLE001
            alpha, beta = 1.0, 0.0
    alpha = float(alpha)
    beta = 0.0 if beta is None else float(beta)
    with np.errstate(divide='ignore', invalid='ignore'):
        G_mass = np.where(r300 > 1e-6, C300 * (1.0 - r300) / np.maximum(r300, 1e-9), 0.0)
    capa = C300 * (300.0 / dt)
    tot = capa + G_mass
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(tot > 0, capa / np.maximum(tot, 1e-9), 0.0)
        g = np.where(tot > 0, G_mass / np.maximum(tot, 1e-9), 0.0)
    L0 = sp.csr_matrix(np.diag(capa + alpha * (Kdir + Kamb) + Mz * (1.0 - g) + Kdiag)
                       - Koff)
    vav_max = np.asarray(m['vav_maxflow'], float)
    min_damp = float(m['min_damp'])
    p_des = float(m['p_design'])
    p_min = float(m['min_static'])
    gmax = vav_max * RHO_CP
    gmin = gmax * min_damp * np.sqrt(p_min / max(p_des, 1.0))
    S = np.asarray(m['share'], float)
    zw = np.asarray(m['zone_w'], float)
    zone_of_node = np.concatenate([np.arange(Nr), np.arange(Nr), [0]])[:nn]
    Amp = np.zeros((Nr, S.shape[1]))
    for i in range(nn):
        Amp[zone_of_node[i]] += S[i]
    zone_vav = np.argmax(Amp, axis=1)
    area = np.asarray(m['cnt'], float)[:Nr] * float(m['cv_area'])
    gain_node = np.zeros(nn)
    for i in range(nn):
        z = zone_of_node[i]
        gain_node[i] = area[z] if z < Nr else 0.0
    return dict(nn=nn, N_room=Nr, capa=capa, r=r, g=g, Kdir=alpha * Kdir,
                Kamb=alpha * Kamb, Mz=Mz, L0=L0, S=S, zw=zw, gmin=gmin, gmax=gmax,
                zone_vav=zone_vav, gain=beta * gain_node, cop=float(m['cop']),
                zone_of_node=zone_of_node,
                tdis=float(m['max_discharge']), recirc=float(m['max_recirc']),
                min_static=float(m['min_static']), fan_eff=float(m['fan_eff']),
                dt=dt, ahu_of_vav=np.asarray(m['ahu_of_vav'], int),
                n_ahu=int(m['A']))


def _schedule(hour0, weekday0, k, dt, start_hour, cm_lo, cm_hi, eco_lo, eco_hi,
              morning=6.0, evening=19.0):
    """Band and comfort flag for slot k, from the clock alone."""
    t = hour0 + k * dt / 3600.0
    hour = t % 24.0
    day = int(weekday0 + (t // 24)) % 7 if weekday0 is not None else 0
    comfort = (day < 5) and (morning <= hour < evening)
    lo = cm_lo if comfort else eco_lo
    hi = cm_hi if comfort else eco_hi
    return lo, hi, comfort


def _solve(m, tzone, outside, hour0, weekday0, cm_lo, cm_hi, eco_lo, eco_hi,
           T, night_floor, cool_gap, flow_cap, start_hour,
           tcold_mode='sup', heat_price_mode='one', plan_margin=0.0):
    nn, Nr = m['nn'], m['N_room']
    dt = m['dt']
    zw = m['zw']
    # state from the observed zone means
    zon = np.repeat(np.arange(Nr), 1)
    x0 = np.zeros(nn)
    zone_of_node = None
    # rooms' core+boundary nodes carry the zone mean; the partition node and
    # the mass nodes take the building mean
    if 'zone_of_node' in m:
        zone_of_node = np.asarray(m['zone_of_node'], int)
    else:
        zone_of_node = np.concatenate([np.arange(Nr), np.arange(Nr), [0]])[:nn]
    mean_all = float(np.mean(tzone))
    for i in range(nn):
        z = zone_of_node[i]
        x0[i] = tzone[z] if z < Nr else mean_all
    Tm0 = x0.copy()
    S = m['S']
    g = m['gmin']
    A = (m['L0'] + sp.csr_matrix(S * g[None, :]) @ sp.csr_matrix(S).T).tocsc()
    Ad = sp.csr_matrix(A).toarray()
    nzA = np.nonzero(Ad)
    capa, r, g_mass, Mz = m['capa'], m['r'], m['g'], m['Mz']
    vent_k = (1.0 - m['recirc']) * RHO_CP
    fan_k = (2.0 - m['recirc']) * m['min_static'] / max(m['fan_eff'], 1e-6)
    tsup = np.full(S.shape[1], m['recirc'] * float(np.mean(tzone))
                   + (1.0 - m['recirc']) * outside)
    tsup_flux = np.asarray(S @ (g * tsup), float).ravel()
    # relative hourly weights (CtrlW's own tables, proportional to the reward)
    try:
        from .ctrl_w import ELEC_REL, GAS_RATIO
    except ImportError:
        from ctrl_w import ELEC_REL, GAS_RATIO
    nvar = 2 * nn + 3 * Nr
    nv = (T + 1) * nvar
    QI, QMI, FI = 2 * nn, 2 * nn + Nr, 2 * nn + 2 * Nr
    # physical mode: the zone, not the mixed-air supply, is the cold side of the
    # reheat coil, so one m3/s delivers RHO_CP*(T_dis - T_zone), not
    # RHO_CP*(T_dis - T_sup), and one watt of zone heat costs the coil
    # 1 + (T_zone - T_sup)/(T_dis - T_zone) watts of gas (the outside-air share
    # at the 308.15 K discharge cap).
    phys_cap = str(tcold_mode) == 'zone'
    phys_price = str(heat_price_mode) == 'phys'
    tz_vec = np.asarray(tzone, float)
    ts_vec = np.array([float(tsup[int(m['zone_vav'][z])]) for z in range(Nr)])
    cap_zone = RHO_CP * np.maximum(m['tdis'] - tz_vec, 1.0) if phys_cap else None
    mult_zone = (1.0 + np.maximum(tz_vec - ts_vec, 0.0)
                 / np.maximum(m['tdis'] - tz_vec, 1.0)) if phys_price else None
    cost = np.zeros(nv)
    bounds = [(None, None)] * nv
    cool_cap = float(np.median(m['gmax'])) * cool_gap
    for k in range(T + 1):
        base = k * nvar
        for i in range(nn):
            bounds[base + i] = (255.0, 330.0)
            bounds[base + nn + i] = (0.0, 0.0) if Mz[i] <= 0 else (255.0, 330.0)
        for z in range(Nr):
            bounds[base + QI + z] = (0.0, 45.0e3)
            bounds[base + QMI + z] = (0.0, cool_cap)
            bounds[base + FI + z] = (0.0, flow_cap)
    for i in range(nn):
        bounds[i] = (float(x0[i]), float(x0[i]))
        bounds[nn + i] = (0.0, 0.0) if Mz[i] <= 0 else (float(Tm0[i]), float(Tm0[i]))
    slo = np.zeros(T); shi = np.zeros(T); scomfort = np.zeros(T, bool)
    for k in range(T):
        lo, hi, cm = _schedule(hour0, weekday0, k, dt, start_hour, cm_lo, cm_hi,
                               eco_lo, eco_hi)
        slo[k], shi[k], scomfort[k] = lo, hi, cm
        hour = int(hour0 + k * dt / 3600.0) % 24
        cg = ELEC_REL[hour] * GAS_RATIO[hour]
        ce = ELEC_REL[hour]
        base = k * nvar
        for z in range(Nr):
            if phys_price:
                cost[base + QI + z] = cg * float(mult_zone[z])
                cost[base + FI + z] = ce * fan_k
            else:
                cost[base + QI + z] = cg
                cost[base + FI + z] = cg * vent_k * max(0.0, float(np.mean(tzone)) - outside) \
                    + ce * fan_k
            cost[base + QMI + z] = ce / m['cop']
    I, J, Vv, BEQ = [], [], [], []
    row = 0
    for k in range(T):
        base = k * nvar
        nxt = base + nvar
        for i, j in zip(*nzA):
            I.append(row + i); J.append(nxt + j); Vv.append(Ad[i, j])
        for i in range(nn):
            I.append(row + i); J.append(base + i); Vv.append(-capa[i])
            I.append(row + i); J.append(base + nn + i); Vv.append(-Mz[i] * r[i])
        for i in range(nn):
            z = zone_of_node[i]
            if z < Nr:
                I.append(row + i); J.append(base + QI + z); Vv.append(-1.0)
                I.append(row + i); J.append(base + QMI + z); Vv.append(1.0)
        force = outside * (m['Kdir'] + m['Kamb']) + tsup_flux + m['gain']
        BEQ.extend(np.asarray(force).ravel()[:nn].tolist())
        row += nn
        for i in range(nn):
            I.append(row + i); J.append(nxt + nn + i); Vv.append(1.0)
            I.append(row + i); J.append(base + nn + i); Vv.append(-r[i])
            I.append(row + i); J.append(nxt + i); Vv.append(-g_mass[i])
            BEQ.append(0.0)
        row += nn
    Aeq = sp.coo_matrix((Vv, (I, J)), shape=(row, nv)).tocsr()
    beq = np.array(BEQ)
    iR, iC, iV, iB = [], [], [], []
    row = 0

    def add(entries, rhs):
        nonlocal row
        for j, v in entries:
            iR.append(row); iC.append(j); iV.append(v)
        iB.append(rhs)
        row += 1

    for k in range(T):
        base = k * nvar
        for z in range(Nr):
            tsup_z = float(tsup[m['zone_vav'][z]])
            cap = (float(cap_zone[z]) if phys_cap
                   else RHO_CP * max(m['tdis'] - tsup_z, 1.0))
            gfl = float(m['gmin'][m['zone_vav'][z]])
            add([(base + QI + z, 1.0), (base + QMI + z, -1.0),
                 (base + FI + z, -cap)], cap * gfl / RHO_CP)
            add([(base + FI + z, 1.0), (base + QI + z, -1.0 / cap),
                 (base + QMI + z, -1.0 / cap)], 0.0)
    Aub = sp.coo_matrix((iV, (iR, iC)), shape=(row, nv)).tocsr()
    bub = np.array(iB)
    cR, cC, cV, cB = [], [], [], []
    row = 0
    for k in range(T):
        base = (k + 1) * nvar
        for z in range(Nr):
            lo = (slo[k] + float(plan_margin)) if scomfort[k] else night_floor
            hi = shi[k] if scomfort[k] else 330.0
            for i in range(nn):
                if zw[i, z]:
                    cR.append(row); cC.append(base + i); cV.append(zw[i, z])
            cB.append(float(hi))
            row += 1
            for i in range(nn):
                if zw[i, z]:
                    cR.append(row); cC.append(base + i); cV.append(-zw[i, z])
            cB.append(-float(lo))
            row += 1
    Aco = sp.coo_matrix((cV, (cR, cC)), shape=(row, nv)).tocsr()
    Aub = sp.vstack([Aub, Aco]).tocsr()
    bub = np.concatenate([bub, np.array(cB)])
    res = linprog(cost, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds,
                  method='highs')
    if not res.success:
        import os as _os
        if _os.environ.get('OPTGAP_MPC_DEBUG'):
            print('MPC LP', res.status, res.message)
        return None
    X = res.x.reshape(T + 1, nvar)
    xz = (X[:T, :nn] @ zw).T          # (n_zones, T)
    return [xz[:, k].copy() for k in range(T)]
