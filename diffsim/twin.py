"""Exact differentiable twin of one simulator step (forward + adjoint).

The twin reproduces ``cenv/src/step.c`` formula by formula -- including the
order of the boiler sensor reads, which is physics on the shipped tree -- and
carries an adjoint for every continuous quantity, so the gradient of the
episode objective with respect to the whole action trajectory costs one more
sparse solve per step.

Working space
-------------
The state lives in the *solved* subspace (``n`` of ``n_full`` control volumes;
the rest are Dirichlet nodes pinned to ambient and never read -- no zone
contains one, and every pinned entry is overwritten by the solve).  The carried
state is

    T      [n]   solved temperatures
    q      [n]   applied power on the solved control volumes
    mass   [n]   interior mass node temperatures
    tank   []    boiler tank temperature
    ret    []    hot-water return temperature (carried while no water flows)

Actions
-------
The action vector is the continuous part of the kernel's command block, in the
order the environment's own continuous action space uses:

    [0, nv)          VAV damper command
    [nv, 2nv)        VAV reheat valve command
    2nv + 0a         AHU a supply-air temperature setpoint
    2nv + na + 0a    AHU a supply-air static pressure setpoint
    2nv + 2na + 0a   AHU a recirculation command
    2nv + 3na        hot-water supply temperature setpoint
    2nv + 3na + 1    pump differential head (m)

The binary run commands are supplied separately (they are not differentiable;
the caller fixes them, normally from a warm-start trajectory).

Convection enters through the per-step sparse permutation precomputed in
``plan.py`` and is applied to the solved destinations only.
"""
from __future__ import annotations

import numpy as np

AIR_DENSITY = 1.2
AIR_CP = 1006.0
WATER_CP = 4180.0
WATER_DENSITY = 1000.0
GRAVITY = 9.8
PASCALS_PER_BAR = 100000.0
FLOW_FLOOR = 1e-5


# --------------------------------------------------------------------------- #
# scalar device maths (ported from cenv/src/hvac.c)
# --------------------------------------------------------------------------- #
def tank_dissipation(water_temp, outside_temp, length, radius, thickness,
                     conductivity, convection):
    if water_temp < outside_temp:
        return 0.0
    delta = water_temp - outside_temp
    numerator = length * 2.0 * np.pi * delta
    r_int = radius
    r_ext = radius + thickness
    conduction = np.log(r_ext / r_int) / conductivity
    conv = 1.0 / convection / r_ext
    return numerator / (conduction + conv)


def tank_temperature(setpoint, actual, duration, heating_rate, cooling_rate):
    if setpoint > actual:
        heated = actual + heating_rate * duration / 60.0
        return heated if heated < setpoint else setpoint
    if setpoint < actual:
        cooled = actual - cooling_rate * duration / 60.0
        return cooled if cooled > setpoint else setpoint
    return setpoint


def _shipped_boiler_rate(supply_setpoint, return_temp, outside_temp, flow,
                         length, radius, thickness, conductivity, convection,
                         tank_change, duration, water_capacity):
    """``Boiler.compute_thermal_energy_rate`` without the energy balance."""
    supply = supply_setpoint if supply_setpoint > return_temp else return_temp
    flow_heat = WATER_CP * WATER_DENSITY * flow * (supply - return_temp)
    dissip = tank_dissipation(supply, outside_temp, length, radius, thickness,
                              conductivity, convection)
    tank_heat = 0.0
    if duration > 0.0:
        tank_heat = (WATER_CP * WATER_DENSITY * water_capacity * tank_change
                     / duration)
    total = flow_heat + dissip + tank_heat
    return total if total > 0.0 else 0.0


def _f32(x):
    return np.float32(x).astype(np.float64)


class Twin:
    def __init__(self, plan: dict, quantize: bool = False):
        """``quantize`` rounds the reward inputs the way the RewardInfo proto
        does (all of its numeric fields are float32).  That is what makes the
        forward gate bit-comparable with the environment's reward; it is off
        for optimisation, where the rounding is a step function and the
        gradient check needs a smooth objective."""
        self.quantize = bool(quantize)
        self.p = plan
        p = plan
        self.n = p["n"]
        self.nz = p["nz"]
        self.nv = p["nv"]
        self.na = p["na"]
        self.k = p["k"]
        self.steps = p["steps"]
        self.dt = p["dt"]
        self.n_act = 2 * self.nv + 3 * self.na + 2
        # geometry
        self.zone_off = p["zone_off"]
        self.zone_cells = p["zone_cells_solved"]
        self.zone_area_f = p["zone_area"].astype(np.float64)
        self.zone_area_i = p["zone_area"]
        self.inv_area = 1.0 / np.maximum(self.zone_area_f, 1.0)
        self.vav_zone = p["vav_zone"]
        self.vav_zoff = p["vav_zoff"]
        self.vav_off = np.concatenate([[0], np.cumsum(p["vav_zn"][:self.nv])]).astype(np.int64)
        if self.vav_zone.size != self.vav_off[-1]:
            raise ValueError("vav_zone length disagrees with the per-VAV zone counts")
        self.vav_zn = p["vav_zn"]
        self.vav_area = p["vav_area"].astype(np.float64)
        self.vav_area_all = p["vav_area_all"].astype(np.float64)
        self.vav_ahu = p["vav_ahu"]
        self.coupled = p["coupled_vav"]
        self.celled = np.zeros(self.n, dtype=bool)
        self.celled[self.zone_cells] = True
        # thermal
        self.cap = p["cap"]
        self.dirichlet = p["dirichlet"]
        self.exposure = p["exposure"]
        self.mc = p["mc"]
        self.mr = p["mr"]
        self.mg = p["mg"]
        self.P = p["P"]
        self.Q = p["Q"]
        self.R = p["R"]
        self.S = p["S"]
        self.factor = p["factor"]
        self.dae = p["dirichlet"] + p["exposure"]
        # tables
        self.ambient = p["ambient"]
        self.comfort = p["comfort"]
        self.occ = p["occ"]
        self.occ_reward = p["occ_reward"]
        self.e_unit = p["e_unit"]
        self.e_carb = p["e_carb"]
        self.g_unit = p["g_unit"]
        self.g_carb = p["g_carb"]
        self.den_e = p["max_e_cost"] + p["max_g_cost"]
        self.den_c = p["max_e_carb"] + p["max_g_carb"]
        self.rp = p["reward"]
        c = p["conv"]
        self.conv_off = c["offsets"]
        self.conv_idx = c["idx"]
        self.conv_src = c["src"]
        # device parameters
        self.vav_max_flow = p["vav_max_flow"]
        self.vav_min_damper = p["vav_min_damper"]
        self.vav_max_discharge = p["vav_max_discharge"]
        self.vav_design = p["vav_design_static"]
        self.vav_reheat_max = p["vav_reheat_max"]
        self.ahu_max_flow = p["ahu_max_flow"]
        self.ahu_min_sp = p["ahu_min_sp"]
        self.ahu_max_recirc = p["ahu_max_recirc"]
        self.ahu_fan_eff = p["ahu_fan_eff"]
        self.ahu_cop = p["ahu_cop"]
        h = p["hws"]
        self.energy_balance = h["energy_balance"]
        self.has_tank = h["has_tank"]
        self.header_resistance = h["header_resistance"]
        self.h_heating_rate = h["heating_rate"]
        self.h_cooling_rate = h["cooling_rate"]
        self.h_conv = h["conv_coef"]
        self.h_len = h["tank_len"]
        self.h_radius = h["tank_radius"]
        self.h_water_cap = h["water_capacity"]
        self.h_insul_cond = h["insul_cond"]
        self.h_insul_thick = h["insul_thick"]
        self.h_pump_eff = h["pump_eff"]
        self.water_thermal_cap = WATER_CP * WATER_DENSITY * self.h_water_cap
        self.internal_gain = p["internal_gain"]
        self.occupant_heat = p["occupant_heat"]
        self.cv_area = p["cv_area"]
        self.comfort_heat = p["comfort_heat"]
        self.comfort_cool = p["comfort_cool"]
        self.eco_heat = p["eco_heat"]
        self.eco_cool = p["eco_cool"]
        self.pinned_count = p["pinned_count"]
        self.n_full = p["n_full"]
        # coupled-column bookkeeping
        self.coupled_ok = self.vav_zn[self.coupled] > 0
        self.coupled_ahu = self.vav_ahu[self.coupled]
        # vav -> zone entries: which (column, zone) pairs a coupled VAV feeds
        col_of = {}
        for c, v in enumerate(self.coupled):
            col_of[int(v)] = c
        ec, ez = [], []
        for v in range(self.nv):
            c = col_of.get(v, -1)
            if c < 0 or self.vav_zn[v] <= 0:
                continue
            for e in range(self.vav_zoff[v], self.vav_zoff[v] + self.vav_zn[v]):
                ec.append(c)
                ez.append(int(self.vav_zone[e]))
        self.qe_col = np.asarray(ec, dtype=np.int64)
        self.qe_zone = np.asarray(ez, dtype=np.int64)
        # zone -> vav entries (for vtemp vjp)
        self.entry_vav = np.repeat(np.arange(self.nv), self.vav_zn)
        self.valid_v = (self.vav_zn > 0) & (self.vav_area > 0)
        self.QT = self.Q.T.tocsr() if self.k else None
        self.PT = self.P.T.tocsr() if self.k else None

    # ------------------------------------------------------------------ #
    def initial_state(self):
        p = self.p
        return dict(T=p["T0"].copy(), q=p["q0"].copy(),
                    mass=p["mass0"].copy(), tank=np.array([p["tank0"]]),
                    ret=np.array([p["ret0"]]))

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def zone_means(self, T):
        g = T[self.zone_cells]
        s = np.add.reduceat(g, self.zone_off[:-1])
        return s * self.inv_area

    def vtemp_of(self, avg):
        if not self.vav_zone.size:
            return np.zeros(self.nv)
        contrib = avg[self.vav_zone] * self.zone_area_f[self.vav_zone]
        vs = np.add.reduceat(contrib, self.vav_off[:-1])
        return np.where(self.valid_v, vs / np.maximum(self.vav_area, 1.0), 0.0)

    def _scatter_avg(self, d_avg, out):
        np.add.at(out, self.zone_cells,
                  np.repeat(d_avg * self.inv_area, self.zone_area_i))
        return out

    def _conv_fwd(self, T, i, amb):
        a, b = self.conv_off[i], self.conv_off[i + 1]
        out = T.copy()
        if b > a:
            idx = self.conv_idx[a:b]
            src = self.conv_src[a:b]
            pose = src >= 0
            out[idx] = np.where(pose, T[np.maximum(src, 0)], amb)
        return out

    def _conv_bwd(self, dout, i):
        """Transpose of the step-i convection map. Returns (dT, d_amb)."""
        a, b = self.conv_off[i], self.conv_off[i + 1]
        dT = dout.copy()
        d_amb = 0.0
        if b > a:
            idx = self.conv_idx[a:b]
            src = self.conv_src[a:b]
            dvals = dout[idx]
            dT[idx] = 0.0
            pose = src >= 0
            np.add.at(dT, src[pose], dvals[pose])
            d_amb = float(dvals[~pose].sum())
        return dT, d_amb

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, acts, runs, state=None, keep_tape=True, i0=0):
        if state is None:
            state = self.initial_state()
        T = state["T"].copy()
        q = state["q"].copy()
        mass = state["mass"].copy()
        tank = float(state["tank"][0])
        ret = float(state["ret"][0])
        tape = [] if keep_tape else None
        nsteps = int(np.asarray(acts).shape[0])
        rewards = np.empty(nsteps)
        penalties = np.empty(nsteps)
        viols = np.empty(nsteps)
        for k in range(nsteps):
            i = i0 + k
            t = self._fwd(i, np.asarray(acts[k], dtype=np.float64),
                          np.asarray(runs[k], dtype=np.float64),
                          T, q, mass, tank, ret)
            T, q, mass, tank, ret = t["out"]
            rewards[k] = t["reward"]
            penalties[k] = t["penalty"]
            viols[k] = t["viol"]
            if keep_tape:
                tape.append(t)
        return rewards, penalties, viols, tape, dict(
            T=T, q=q, mass=mass, tank=np.array([tank]), ret=np.array([ret]))

    def _fwd(self, i, act, run, T, q, mass, tank, ret):
        nv, na, k, nz = self.nv, self.na, self.k, self.nz
        a_d = act[:nv]
        a_v = act[nv:2 * nv]
        o1 = 2 * nv
        a_sp = act[o1:o1 + na]
        a_fp = act[o1 + na:o1 + 2 * na]
        a_rc = act[o1 + 2 * na:o1 + 3 * na]
        hws_sp = float(act[o1 + 3 * na])
        pump_head = float(act[o1 + 3 * na + 1])
        run_a = run[:na]
        pump_run = run[na]

        amb = self.ambient[i]
        amb1 = self.ambient[i + 1]
        comfort = bool(self.comfort[i])
        comfort1 = bool(self.comfort[i + 1])

        avg = self.zone_means(T)
        vtemp = self.vtemp_of(avg)

        damper = np.maximum(a_d, self.vav_min_damper)
        valve = a_v.copy()
        press = np.where(run_a > 0, np.maximum(a_fp, self.ahu_min_sp), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ffac = np.sqrt(np.maximum(press[self.vav_ahu], 0.0) / self.vav_design)
        uraw = damper * self.vav_max_flow * ffac
        unrestr = np.where(uraw > FLOW_FLOOR, uraw, FLOW_FLOOR)

        if self.energy_balance:
            uv = np.where((damper > 0.0) & (self.vav_max_flow > 0.0)
                          & self.valid_v, unrestr, 0.0)
            fsum_r = np.bincount(self.vav_ahu, weights=vtemp * uv, minlength=na)
            ftot_r = np.bincount(self.vav_ahu, weights=uv, minlength=na)
        else:
            uv = np.zeros(nv)
            fsum_r = np.zeros(na)
            ftot_r = np.zeros(na)
        asum_r = np.bincount(self.vav_ahu,
                             weights=np.where(self.valid_v, vtemp * self.vav_area, 0.0),
                             minlength=na)
        atot_r = np.bincount(self.vav_ahu,
                             weights=np.where(self.valid_v, self.vav_area, 0.0),
                             minlength=na)
        flat = float(T.sum() + self.pinned_count * amb) / self.n_full
        with np.errstate(divide="ignore", invalid="ignore"):
            recirc = np.where(ftot_r > 0.0, fsum_r / np.maximum(ftot_r, 1e-300),
                              np.where(atot_r > 0.0, asum_r / np.maximum(atot_r, 1e-300),
                                       flat))
        mixed = a_rc * recirc + (1.0 - a_rc) * amb
        sw = (mixed > a_sp) & (run_a > 0)
        supply = np.where(sw, a_sp, mixed)

        fsum = float(np.sum(valve * self.vav_reheat_max))
        dp = (GRAVITY * WATER_DENSITY * pump_head / PASCALS_PER_BAR) if pump_run > 0 else 0.0
        hw_res = 1.0 + fsum * fsum * self.header_resistance
        sflow = np.sqrt(max(dp, 0.0) / hw_res)
        total_flow = fsum * sflow / 3600.0
        w_cap = WATER_CP * WATER_DENSITY * total_flow

        demand = np.bincount(self.vav_ahu, weights=unrestr, minlength=na)
        sat = demand > self.ahu_max_flow
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(demand > 0.0,
                             np.minimum(1.0, self.ahu_max_flow / np.maximum(demand, 1e-300)),
                             1.0)
        flow = unrestr * ratio[self.vav_ahu]

        g = np.zeros(k)
        ts = np.zeros(k)
        reheat = np.zeros(k)
        rf = np.zeros(k)
        eff = np.zeros(k)
        gate = np.zeros(k, dtype=bool)
        if k:
            v = self.coupled
            f = flow[v]
            sup = supply[self.coupled_ahu]
            air_cap = AIR_CP * AIR_DENSITY * f
            with np.errstate(divide="ignore", invalid="ignore"):
                rf = np.where(fsum != 0.0,
                              total_flow * (valve[v] * self.vav_reheat_max[v]
                                            / (fsum if fsum != 0.0 else 1.0)),
                              0.0)
            water_cap = WATER_CP * WATER_DENSITY * rf
            gap = tank - sup
            headroom = self.vav_max_discharge[v] - sup
            with np.errstate(divide="ignore", invalid="ignore"):
                e = np.where(air_cap > 0.0,
                             np.minimum(water_cap / np.where(air_cap > 0.0, air_cap, 1.0), 1.0),
                             0.0)
                cap_e = np.where(gap > 0.0, headroom / np.where(gap > 0.0, gap, 1.0), 0.0)
            effc = np.where(air_cap <= 0.0, 0.0,
                            np.where(gap <= 0.0, e,
                                     np.where(headroom <= 0.0, 0.0, np.minimum(cap_e, e))))
            reheat_c = effc * AIR_CP * AIR_DENSITY * f * (tank - sup)
            gate_c = (damper[v] > 0.0) & (self.vav_max_flow[v] > 0.0) & self.coupled_ok
            g = np.where(gate_c, f * AIR_DENSITY * AIR_CP, 0.0)
            gate = gate_c
            ts = np.where(self.coupled_ok, sup, 0.0)
            reheat = np.where(self.coupled_ok, reheat_c, 0.0)
            rf = np.where(self.coupled_ok, rf, 0.0)
            eff = np.where(self.coupled_ok, effc, 0.0)
        total_reheat = float(np.sum(reheat))
        coil = ts[self.coupled_ok]
        coil_n = int(coil.size)
        coil_min = float(coil.min()) if coil_n else 0.0
        coil_max = float(coil.max()) if coil_n else 0.0

        # zone loads
        qz = np.zeros(nz)
        if self.qe_col.size:
            v = self.coupled
            share = np.where(gate[self.qe_col],
                             reheat[self.qe_col] / np.maximum(self.vav_area[v[self.qe_col]], 1.0),
                             0.0)
            np.add.at(qz, self.qe_zone, share * self.zone_area_f[self.qe_zone])
        watts_per_cv = self.internal_gain * self.cv_area * (1.0 if comfort else 0.2)
        qz += watts_per_cv * self.zone_area_f
        occ = self.occ[i]
        qz += occ * self.occupant_heat
        q_solved = q.copy()
        q_solved[self.zone_cells] = np.repeat(qz * self.inv_area, self.zone_area_i)

        # thermal
        fvec = self.cap * T + q_solved
        fvec = fvec + amb * self.dae
        fvec = fvec + self.mc * mass
        y = self.factor.solve(fvec)
        if k:
            sy = self.Q @ y
            b = g * (ts - sy)
            Bn = np.eye(k) + g[:, None] * self.S
            rhs = np.linalg.solve(Bn, b)
            Tp = y + self.R @ rhs
        else:
            sy = np.zeros(0)
            b = np.zeros(0)
            Bn = np.zeros((0, 0))
            rhs = np.zeros(0)
            Tp = y
        mass_new = self.mr * mass + self.mg * Tp

        Tpost = self._conv_fwd(Tp, i, amb)

        # epilogue: return water and tank
        if w_cap > 0.0 and coil_n > 0:
            ret_raw = tank - total_reheat / w_cap
            air_temp = coil_min if total_reheat >= 0.0 else coil_max
            lo = min(tank, air_temp)
            hi = max(tank, air_temp)
            ret_new = min(max(ret_raw, lo), hi)
            ret_active = True
        else:
            ret_raw = ret
            ret_new = ret
            ret_active = False
            lo = hi = air_temp = 0.0
        coil_heat = w_cap * (tank - ret_new)
        if self.energy_balance:
            # the tank balance uses *this* step's ambient (the epilogue's
            # `ambient`, not `ambient1`)
            loss = tank_dissipation(tank, amb, self.h_len, self.h_radius,
                                    self.h_insul_thick, self.h_insul_cond, self.h_conv)
            needed = self.water_thermal_cap * (hws_sp - tank) / self.dt + coil_heat + loss
            max_rate = (coil_heat if coil_heat > 0.0 else 0.0) + loss \
                + self.water_thermal_cap * self.h_heating_rate / 60.0
            burner = needed if needed > 0.0 else 0.0
            clamped = burner > max_rate
            if clamped:
                burner = max_rate
            tank_new = tank + (burner - coil_heat - loss) * self.dt / self.water_thermal_cap
        else:
            tank_obs2 = tank_temperature(hws_sp, tank, self.dt,
                                         self.h_heating_rate, self.h_cooling_rate)
            step_change = tank_obs2 - tank
            burner = _shipped_boiler_rate(
                hws_sp, ret_new, amb1, total_flow, self.h_len, self.h_radius,
                self.h_insul_thick, self.h_insul_cond, self.h_conv,
                step_change, self.dt, self.h_water_cap)
            tank_new = tank_obs2
            loss = 0.0
            needed = 0.0
            max_rate = 0.0
            clamped = False

        # post-step plant rates
        avg2 = self.zone_means(Tpost)
        vtemp2 = self.vtemp_of(avg2)
        if self.energy_balance:
            uv2 = np.where((damper > 0.0) & (self.vav_max_flow > 0.0)
                           & self.valid_v, unrestr, 0.0)
            fsum2 = np.bincount(self.vav_ahu, weights=vtemp2 * uv2, minlength=na)
            ftot2 = np.bincount(self.vav_ahu, weights=uv2, minlength=na)
        else:
            uv2 = np.zeros(nv)
            fsum2 = np.zeros(na)
            ftot2 = np.zeros(na)
        asum2 = np.bincount(self.vav_ahu,
                            weights=np.where(self.valid_v, vtemp2 * self.vav_area, 0.0),
                            minlength=na)
        atot2 = np.bincount(self.vav_ahu,
                            weights=np.where(self.valid_v, self.vav_area, 0.0),
                            minlength=na)
        flat2 = float(Tpost.sum() + self.pinned_count * amb) / self.n_full
        with np.errstate(divide="ignore", invalid="ignore"):
            recirc2 = np.where(ftot2 > 0.0, fsum2 / np.maximum(ftot2, 1e-300),
                               np.where(atot2 > 0.0, asum2 / np.maximum(atot2, 1e-300),
                                        flat2))
        airflow = np.bincount(self.vav_ahu, weights=flow, minlength=na)
        airflow_eff = np.where(run_a > 0, airflow, 0.0)
        sp_eff = np.where(run_a > 0, np.maximum(a_fp, self.ahu_min_sp), 0.0)
        mixed2 = a_rc * recirc2 + (1.0 - a_rc) * amb1
        sw2 = (mixed2 > a_sp) & (run_a > 0)
        supply2 = np.where(sw2, a_sp, mixed2)
        intake = airflow_eff * sp_eff / self.ahu_fan_eff
        exhaust = airflow_eff * (1.0 - a_rc) * sp_eff / self.ahu_fan_eff
        cooling = airflow_eff * AIR_DENSITY * AIR_CP * (supply2 - mixed2) / self.ahu_cop
        head = pump_head if pump_run > 0 else 0.0
        pump_w = total_flow * WATER_DENSITY * GRAVITY * head / self.h_pump_eff

        # reward
        rp = self.rp
        dt = self.dt
        hs = self.comfort_heat if comfort1 else self.eco_heat
        cs = self.comfort_cool if comfort1 else self.eco_cool
        # the reward's occupancy is the simulator's own Python chain (see
        # plan.py); the lids/gains above use the kernel's copy
        occ2 = self.occ_reward[i + 1]
        if self.quantize:
            tz = _f32(avg2)
            hs_q = _f32(hs)
            cs_q = _f32(cs)
            occ2_q = _f32(occ2)
        else:
            tz, hs_q, cs_q, occ2_q = avg2, hs, cs, occ2
        dev_c = np.maximum(0.0, tz - cs_q)
        dev_h = np.maximum(0.0, hs_q - tz)
        dev = np.maximum(dev_c, dev_h)
        ref = rp["deviation_ref"]
        short = np.where(dev <= ref, dev / ref, 1.0 + ((dev - ref) / ref) ** 2)
        prod = rp["max_productivity"] * (1.0 - short)
        cum = float(np.sum(prod * occ2_q * dt / 3600.0))
        occ_tot = float(occ2_q.sum())
        maxp_tot = rp["max_productivity"] * occ_tot * dt / 3600.0
        minp_tot = rp["min_productivity"] * occ_tot * dt / 3600.0
        nreg = (cum - minp_tot) / (maxp_tot - minp_tot) - 1.0 if occ_tot > 0.0 else 0.0
        # the reward's AHU "blower" rate is intake + exhaust (cenv's
        # build_reward_info sums both into blower_electrical_energy_rate)
        if self.quantize:
            blower = float(_f32(float(np.sum(intake)) + float(np.sum(exhaust))))
            ac = float(_f32(float(np.sum(np.abs(cooling)))))
            pump_q = float(_f32(pump_w))
            gas = float(_f32(burner))
        else:
            blower = float(np.sum(intake) + np.sum(exhaust))
            ac = float(np.sum(np.abs(cooling)))
            pump_q = float(pump_w)
            gas = float(burner)
        elec = blower + ac + pump_q
        # the reward info's window is [t_{i+1}, t_{i+2}): the simulator builds
        # it at the post-step timestamp, so the price/carbon index is i + 1
        j = i + 1
        ecost = self.e_unit[j] * abs(elec)
        gcost = self.g_unit[j] * (gas if gas > 0.0 else 0.0)
        ecarb = self.e_carb[j] * abs(elec)
        gcarb = self.g_carb[j] * (gas if gas > 0.0 else 0.0)
        nenergy = (ecost + gcost) / self.den_e[j] if self.den_e[j] > 0.0 else 0.0
        ncarb = (ecarb + gcarb) / self.den_c[j] if self.den_c[j] > 0.0 else 0.0
        W = rp["w_p"] + rp["w_e"] + rp["w_c"]
        raw = rp["w_p"] * nreg - rp["w_e"] * nenergy - rp["w_c"] * ncarb
        reward = raw / W
        if self.quantize:
            # RewardResponse.agent_reward_value is a proto `float`
            reward = float(_f32(reward))
        viol = float(np.maximum(dev_c, dev_h).max()) if nz else 0.0
        penalty = float(np.sum(dev_c * dev_c + dev_h * dev_h)) if comfort1 else 0.0

        return dict(
            i=i, act=act, run=run, amb=amb, amb1=amb1, comfort=comfort,
            comfort1=comfort1, avg=avg, vtemp=vtemp, damper=damper, valve=valve,
            press=press, ffac=ffac, uraw=uraw, unrestr=unrestr, uv=uv,
            fsum_r=fsum_r, ftot_r=ftot_r, asum_r=asum_r, atot_r=atot_r,
            recirc=recirc, mixed=mixed, supply=supply, sw=sw, fsum=fsum, dp=dp,
            hw_res=hw_res, sflow=sflow, total_flow=total_flow, w_cap=w_cap,
            demand=demand, sat=sat, ratio=ratio, flow=flow, g=g, ts=ts,
            reheat=reheat, rf=rf, eff=eff, gate=gate, total_reheat=total_reheat,
            coil_min=coil_min, coil_max=coil_max, coil_n=coil_n,
            ret=ret, ret_new=ret_new, ret_raw=ret_raw, ret_active=ret_active,
            lo=lo, hi=hi, air_temp=air_temp, coil_heat=coil_heat, loss=loss,
            needed=needed, max_rate=max_rate, clamped=clamped, burner=burner,
            tank=tank, tank_new=tank_new,
            avg2=avg2, vtemp2=vtemp2, uv2=uv2, fsum2=fsum2, ftot2=ftot2,
            asum2=asum2, atot2=atot2, recirc2=recirc2, airflow=airflow,
            airflow_eff=airflow_eff, sp_eff=sp_eff, mixed2=mixed2, sw2=sw2,
            supply2=supply2, intake=intake, exhaust=exhaust, cooling=cooling,
            pump_w=pump_w, head=head, hws_sp=hws_sp, pump_head=pump_head,
            pump_run=float(pump_run),
            fvec=fvec, y=y, sy=sy, b=b, Bn=Bn, rhs=rhs, Tp=Tp,
            mass_new=mass_new, q_solved=q_solved, qz=qz, occ=occ,
            dev_c=dev_c, dev_h=dev_h, dev=dev, short=short, prod=prod,
            occ2=occ2, occ_tot=occ_tot, maxp_tot=maxp_tot, cum=cum,
            elec=elec, gas=gas, reward=reward, viol=viol, penalty=penalty,
            T=T, mass=mass, q=q,
            out=(Tpost, q_solved, mass_new, tank_new, ret_new),
        )

    # ------------------------------------------------------------------ #
    # backward
    # ------------------------------------------------------------------ #
    def backward(self, tape, d_reward=None, penalty_mu=0.0):
        """dJ/d(actions) and dJ/d(state0) for J = sum(dr*r) + mu * sum(pen).

        The reverse pass mirrors ``_fwd`` block for block.  Gradients with
        respect to a *constant* quantity (ambient, occupancy, prices) are not
        carried anywhere: those are data, not decisions.
        """
        nv, na, k, nz, n = self.nv, self.na, self.k, self.nz, self.n
        steps = len(tape)
        d_acts = np.zeros((steps, self.n_act))
        if d_reward is None:
            d_reward = np.ones(steps)
        dT = np.zeros(n)
        dM = np.zeros(n)
        dq = np.zeros(n)
        dtank = np.zeros(1)
        dret = np.zeros(1)
        for i in range(steps - 1, -1, -1):
            t = tape[i]
            dr = float(d_reward[i])
            da = d_acts[i]
            rp = self.rp
            W = rp["w_p"] + rp["w_e"] + rp["w_c"]
            nv, na, k = self.nv, self.na, self.k

            # ---------- reward -----------------------------------------
            d_avg2 = np.zeros(nz)
            d_avg = np.zeros(nz)
            d_elec = 0.0
            d_gas = 0.0
            if dr != 0.0 or penalty_mu:
                dnreg = dr * rp["w_p"] / W
                dnenergy = -dr * rp["w_e"] / W
                dncarb = -dr * rp["w_c"] / W
                den = t["maxp_tot"] - rp["min_productivity"] * t["occ_tot"] * self.dt / 3600.0
                d_cum = dnreg / den if den != 0.0 else 0.0
                # the price/carbon index is the ABSOLUTE step index (the tape
                # may start at i0 > 0); the forward used j = t["i"] + 1
                j = int(t["i"]) + 1
                dd_e = dnenergy / self.den_e[j] if self.den_e[j] > 0.0 else 0.0
                dd_c = dncarb / self.den_c[j] if self.den_c[j] > 0.0 else 0.0
                sign_e = 1.0 if t["elec"] >= 0.0 else -1.0
                d_elec = dd_e * self.e_unit[j] * sign_e + dd_c * self.e_carb[j] * sign_e
                if t["gas"] > 0.0:
                    d_gas = dd_e * self.g_unit[j] + dd_c * self.g_carb[j]
                ref = rp["deviation_ref"]
                dshort = -rp["max_productivity"] * (d_cum * t["occ2"] * self.dt / 3600.0)
                ddev = np.where(t["dev"] <= ref, dshort / ref,
                                dshort * 2.0 * (t["dev"] - ref) / (ref * ref))
                hi_m = (t["dev_c"] >= t["dev_h"]) & (t["dev_c"] > 0.0)
                lo_m = (t["dev_h"] > t["dev_c"]) & (t["dev_h"] > 0.0)
                d_avg2 += np.where(hi_m, ddev, 0.0)
                d_avg2 -= np.where(lo_m, ddev, 0.0)
                if penalty_mu:
                    d_avg2 += penalty_mu * (2.0 * t["dev_c"] - 2.0 * t["dev_h"])

            # ---------- plant rates at the reward ----------------------
            d_intake = np.full(na, d_elec)
            d_exhaust = np.full(na, d_elec)
            d_cooling = d_elec * np.sign(t["cooling"])
            d_pump_w = d_elec
            fac = AIR_DENSITY * AIR_CP
            d_airflow_eff = d_cooling / self.ahu_cop * fac * (t["supply2"] - t["mixed2"])
            d_supply2 = d_cooling / self.ahu_cop * t["airflow_eff"] * fac
            d_mixed2 = -d_supply2
            da_sp = np.where(t["sw2"], d_supply2, 0.0)
            d_mixed2 = np.where(t["sw2"], 0.0, d_mixed2)
            rc = t["act"][2 * nv + 2 * na:2 * nv + 3 * na]
            d_recirc2 = d_mixed2 * rc
            d_rc = d_mixed2 * (t["recirc2"] - t["amb1"])
            sp_eff = t["sp_eff"]
            afe = t["airflow_eff"]
            d_airflow_eff += d_intake * sp_eff / self.ahu_fan_eff
            d_airflow_eff += d_exhaust * (1.0 - rc) * sp_eff / self.ahu_fan_eff
            d_sp_eff = d_intake * afe / self.ahu_fan_eff \
                + d_exhaust * afe * (1.0 - rc) / self.ahu_fan_eff
            d_rc += d_exhaust * (-afe * sp_eff / self.ahu_fan_eff)
            d_airflow = np.where(t["run"][:na] > 0, d_airflow_eff, 0.0)
            # pump power = total_flow*rho*g*head/eff
            head = t["head"]
            d_total_flow = d_pump_w * WATER_DENSITY * GRAVITY * head / self.h_pump_eff
            d_head = d_pump_w * t["total_flow"] * WATER_DENSITY * GRAVITY / self.h_pump_eff

            # ---------- recirc2 -> vtemp2 (+ unrestr), flow ------------
            d_vtemp2 = np.zeros(nv)
            d_unrestr = np.zeros(nv)
            for a in range(na):
                m = self.vav_ahu == a
                if t["ftot2"][a] > 0.0:
                    d_vtemp2 += d_recirc2[a] * t["uv2"] / t["ftot2"][a] * m
                    d_unrestr += d_recirc2[a] * (t["vtemp2"] - t["recirc2"][a]) \
                        / t["ftot2"][a] * m
                elif t["atot2"][a] > 0.0:
                    d_vtemp2 += d_recirc2[a] * self.vav_area / t["atot2"][a] * m
                else:
                    dT += d_recirc2[a] / self.n_full
            # uv2 gate: (damper>0)&(max_flow>0)&valid
            d_unrestr *= ((t["damper"] > 0.0) & (self.vav_max_flow > 0.0)
                          & self.valid_v)
            # airflow = min(sum_v flow, max_flow) -> passes to each flow only
            # when the sum was not clipped
            d_flow = np.zeros(nv)
            uncapped = t["airflow"] < self.ahu_max_flow
            for a in range(na):
                if uncapped[a]:
                    d_flow[self.vav_ahu == a] += d_airflow[a]
            # vtemp2 is derived from the POST-step zone means, so its adjoint
            # belongs on the post-convection field, not on the step-start one
            d_avg2 += self._vtemp_vjp(d_vtemp2)

            # ---------- convection + reward scatter -------------------
            dtf_post = np.zeros(n)
            self._scatter_avg(d_avg2, dtf_post)
            d_pre_conv, _ = self._conv_bwd(dT + dtf_post, i)

            # ---------- thermal ---------------------------------------
            dTp = d_pre_conv + self.mg * dM
            dM_prev = self.mr * dM
            if k:
                drhs = self.R.T @ dTp
                db = np.linalg.solve(t["Bn"].T, drhs)
                dBn = -np.outer(db, t["rhs"])
                dg = np.sum(dBn * self.S, axis=1) + db * (t["ts"] - t["sy"])
                dts = db * t["g"]
                dsy = -db * t["g"]
                dy = dTp + self.QT @ dsy
            else:
                dg = np.zeros(0)
                dts = np.zeros(0)
                dy = dTp
            df = self.factor.solve(dy)
            dT_prev = self.cap * df
            dM_prev = dM_prev + self.mc * df
            dqs = df + dq          # fvec + carried state
            dq_prev = np.where(self.celled, 0.0, dqs)
            dqz = np.add.reduceat(dqs[self.zone_cells], self.zone_off[:-1]) * self.inv_area

            # ---------- epilogue (tank / return water / burner) -------
            epi = self._epilogue_bwd(t, d_gas, float(dtank[0]), float(dret[0]),
                                     d_total_flow, d_head)
            d_total_flow = epi["d_total_flow"]
            d_head = epi["d_head"]
            d_tank = epi["d_tank"]
            dret_prev = epi["dret_prev"]
            d_total_reheat = epi["d_total_reheat"]
            d_hws_sp = epi["d_hws_sp"]
            d_coil_min = epi["d_coil_min"]
            d_coil_max = epi["d_coil_max"]
            d_ts = dts.copy()
            if t["coil_n"]:
                vals = t["ts"][self.coupled_ok]
                if d_coil_min != 0.0:
                    j = int(np.argmin(vals))
                    d_ts[self.coupled[self.coupled_ok]][j] += d_coil_min
                if d_coil_max != 0.0:
                    j = int(np.argmax(vals))
                    d_ts[self.coupled[self.coupled_ok]][j] += d_coil_max

            # ---------- zone loads -> reheat --------------------------
            d_reheat = np.zeros(k)
            if self.qe_col.size:
                v = self.coupled
                sh = dqz[self.qe_zone] * self.zone_area_f[self.qe_zone] \
                    / np.maximum(self.vav_area[v[self.qe_col]], 1.0)
                np.add.at(d_reheat, self.qe_col,
                          np.where(t["gate"][self.qe_col], sh, 0.0))
            d_reheat += d_total_reheat

            # ---------- g = flow*rho*cp on the gated columns ----------
            if k:
                d_flow += np.where(t["gate"], dg * AIR_DENSITY * AIR_CP, 0.0)
            # ---------- reheat ----------------------------------------
            rh = self._reheat_bwd(t, d_reheat, d_ts, d_tank)
            d_flow += rh["d_flow"]
            d_valve = rh["d_valve"]
            d_fsum = rh["d_fsum"]
            d_tank = rh["d_tank"]
            d_supply = rh["d_supply"]
            d_total_flow += rh["d_total_flow"]

            # ---------- supply / mixed at the step start --------------
            sw = t["sw"]
            d_mixed = np.where(sw, 0.0, d_supply)
            da_sp += np.where(sw, d_supply, 0.0)
            d_recirc = d_mixed * rc
            d_rc += d_mixed * (t["recirc"] - t["amb"])

            # ---------- recirculation at the step start ---------------
            if self.energy_balance:
                d_uv, d_vtemp = self._recirc_bwd(t, d_recirc)
                d_unrestr += d_uv * ((t["damper"] > 0.0) & (self.vav_max_flow > 0.0)
                                     & self.valid_v)
            else:
                d_vtemp = self._area_recirc_bwd(t, d_recirc)
            d_avg += self._vtemp_vjp(d_vtemp)

            # ---------- flow cap --------------------------------------
            d_unrestr += d_flow * t["ratio"][self.vav_ahu]
            d_ratio = d_flow * t["unrestr"]
            d_demand = np.zeros(na)
            for a in range(na):
                if t["sat"][a] and t["demand"][a] > 0.0:
                    m = self.vav_ahu == a
                    d_demand[a] = -np.sum(d_ratio[m]) * self.ahu_max_flow[a] \
                        / t["demand"][a] ** 2
            d_unrestr += d_demand[self.vav_ahu]

            # ---------- unrestricted flow -----------------------------
            d_uraw = d_unrestr * (t["uraw"] > FLOW_FLOOR)
            d_damper = d_uraw * self.vav_max_flow * t["ffac"]
            d_ffac = d_uraw * t["damper"] * self.vav_max_flow
            d_press = np.zeros(na)
            okf = t["ffac"] > 0.0
            d_press[self.vav_ahu[okf]] += (d_ffac[okf] * 0.5
                                           / (t["ffac"][okf] * self.vav_design[okf]))
            d_press += d_sp_eff

            # ---------- hot-water flow (fsum, dp) ---------------------
            d_fsum2, d_dp = self._flow_bwd(t, d_total_flow, d_fsum)
            d_valve = d_valve + d_fsum2 * self.vav_reheat_max
            d_pump_head = 0.0
            if t["pump_run"] > 0:
                d_pump_head = d_dp * GRAVITY * WATER_DENSITY / PASCALS_PER_BAR

            # ---------- actions ---------------------------------------
            o1 = 2 * nv
            da[:nv] += np.where(t["act"][:nv] > self.vav_min_damper, d_damper, 0.0)
            da[nv:2 * nv] += d_valve
            da[o1:o1 + na] += da_sp
            da[o1 + na:o1 + 2 * na] += np.where(
                (t["run"][:na] > 0) & (t["act"][o1 + na:o1 + 2 * na] > self.ahu_min_sp),
                d_press, 0.0)
            da[o1 + 2 * na:o1 + 3 * na] += d_rc
            da[o1 + 3 * na] += d_hws_sp
            da[o1 + 3 * na + 1] += d_pump_head

            # ---------- carried state ---------------------------------
            dT = dT_prev
            dT = dT + self._scatter_avg(d_avg, np.zeros(n))
            dM = dM_prev
            dq = dq_prev
            dtank = np.array([d_tank])
            dret = dret_prev
        return d_acts, dict(T=dT, q=dq, mass=dM, tank=dtank, ret=dret)

    # ------------------------------------------------------------------ #
    # block VJPs
    # ------------------------------------------------------------------ #
    def _vtemp_vjp(self, d_vtemp):
        d_avg = np.zeros(self.nz)
        if not self.vav_zone.size:
            return d_avg
        e = self.entry_vav
        good = self.valid_v[e]
        contrib = np.where(good, d_vtemp[e] * self.zone_area_f[self.vav_zone]
                           / np.maximum(self.vav_area[e], 1.0), 0.0)
        np.add.at(d_avg, self.vav_zone, contrib)
        return d_avg

    def _recirc_bwd(self, t, d_recirc):
        """d_uv, d_vtemp for the flow-weighted (corrected) recirculation."""
        nv, na = self.nv, self.na
        d_uv = np.zeros(nv)
        d_vtemp = np.zeros(nv)
        for a in range(na):
            m = self.vav_ahu == a
            if t["ftot_r"][a] > 0.0:
                d_uv += d_recirc[a] * (t["vtemp"] - t["recirc"][a]) / t["ftot_r"][a] * m
                d_vtemp += d_recirc[a] * t["uv"] / t["ftot_r"][a] * m
            elif t["atot_r"][a] > 0.0:
                d_vtemp += d_recirc[a] * self.vav_area / t["atot_r"][a] * m
        return d_uv, d_vtemp

    def _area_recirc_bwd(self, t, d_recirc):
        d_vtemp = np.zeros(self.nv)
        for a in range(self.na):
            m = self.vav_ahu == a
            if t["atot_r"][a] > 0.0:
                d_vtemp += d_recirc[a] * self.vav_area / t["atot_r"][a] * m
        return d_vtemp

    def _flow_bwd(self, t, d_total_flow, d_fsum_in):
        d_fsum = d_fsum_in
        d_dp = 0.0
        sflow = t["sflow"]
        d_sflow = d_total_flow * t["fsum"] / 3600.0
        d_fsum += d_total_flow * sflow / 3600.0
        if sflow > 0.0:
            d_dp = d_sflow * 0.5 / sflow
            d_hw_res = -d_sflow * sflow / (2.0 * t["hw_res"])
            d_fsum += d_hw_res * 2.0 * t["fsum"] * self.header_resistance
        return d_fsum, d_dp

    def _reheat_bwd(self, t, d_reheat, d_ts, d_tank):
        """Reheat coils, their water/air capacity rates and the supply temps."""
        k = self.k
        d_flow = np.zeros(self.nv)
        d_valve = np.zeros(self.nv)
        d_fsum = 0.0
        d_supply = np.zeros(self.na)
        if not k:
            return dict(d_flow=d_flow, d_valve=d_valve, d_fsum=d_fsum,
                        d_tank=d_tank, d_supply=d_supply)
        v = self.coupled
        ok = self.coupled_ok
        dr = d_reheat * ok
        f = t["flow"][v]
        sup = t["supply"][self.coupled_ahu]
        tank = t["tank"]
        cp = AIR_CP * AIR_DENSITY
        air_cap = cp * f
        gap = tank - sup
        headroom = self.vav_max_discharge[v] - sup
        eff = t["eff"]
        rf = t["rf"]
        total_flow = t["total_flow"]
        fsum = t["fsum"]
        # reheat = eff*cp*f*(tank-sup)
        d_eff = dr * cp * f * (tank - sup)
        d_f = dr * eff * cp * (tank - sup)
        d_tank_c = dr * eff * cp * f
        d_sup = -dr * eff * cp * f
        # eff derivative
        air_ok = air_cap > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            e = np.where(air_ok, np.minimum(t["rf"] * WATER_CP * WATER_DENSITY
                                            / np.where(air_ok, air_cap, 1.0), 1.0), 0.0)
            cap_e = np.where(gap > 0.0, headroom / np.where(gap > 0.0, gap, 1.0), 0.0)
        not_clamped = air_ok & ((t["rf"] * WATER_CP * WATER_DENSITY) < air_cap)
        d_e = np.zeros(k)
        d_cap = np.zeros(k)
        branch_e = air_ok & (gap <= 0.0)
        branch_min = air_ok & (gap > 0.0) & (headroom > 0.0)
        cap_lt = branch_min & (cap_e < e)
        d_e += np.where(branch_e, d_eff, 0.0)
        d_e += np.where(branch_min & ~cap_lt, d_eff, 0.0)
        d_cap += np.where(cap_lt, d_eff, 0.0)
        water_cap = t["rf"] * WATER_CP * WATER_DENSITY
        d_water_cap = np.where(not_clamped, d_e * ok / np.where(air_ok, air_cap, 1.0), 0.0)
        d_air_cap = np.where(not_clamped,
                             -d_e * water_cap / np.where(air_ok, air_cap, 1.0) ** 2, 0.0)
        d_cap_w = np.where(branch_min, d_cap, 0.0)
        d_gap = np.where(branch_min, -d_cap_w * headroom / np.where(gap > 0.0, gap, 1.0) ** 2, 0.0)
        d_headroom = np.where(branch_min, d_cap_w / np.where(gap > 0.0, gap, 1.0), 0.0)
        d_tank_c += d_gap
        d_sup += -d_gap - d_headroom
        d_f += d_air_cap * cp
        d_rf = d_water_cap * WATER_CP * WATER_DENSITY
        if fsum != 0.0:
            d_total_flow = np.sum(d_rf * (t["valve"][v] * self.vav_reheat_max[v] / fsum))
            np.add.at(d_valve, v, d_rf * total_flow * self.vav_reheat_max[v] / fsum)
            d_fsum += -np.sum(d_rf * total_flow * t["valve"][v]
                              * self.vav_reheat_max[v] / fsum ** 2)
        else:
            d_total_flow = 0.0
        d_flow[v] += d_f
        # ts[c] = supply[ahu(c)]: the air-side gradient on ts, and the coil
        # min/max selector, both land on the same handler temperatures.
        np.add.at(d_supply, self.coupled_ahu, d_sup + np.where(ok, d_ts, 0.0))
        return dict(d_flow=d_flow, d_valve=d_valve, d_fsum=d_fsum,
                    d_tank=d_tank + float(np.sum(d_tank_c)),
                    d_supply=d_supply, d_total_flow=d_total_flow)

    def _epilogue_bwd(self, t, d_gas, dtank_in, dret_in, d_total_flow, d_head):
        dt = self.dt
        Cw = self.water_thermal_cap
        d_tank = 0.0
        d_hws_sp = 0.0
        d_total_reheat = 0.0
        d_coil_min = 0.0
        d_coil_max = 0.0
        d_wcap = 0.0
        d_total_flow = d_total_flow
        d_head = d_head
        if self.energy_balance:
            d_burner = d_gas
            d_coil = -dtank_in * dt / Cw
            d_loss = -dtank_in * dt / Cw
            d_tank += dtank_in
            d_needed = d_burner * (1.0 if (t["needed"] > 0.0 and not t["clamped"]) else 0.0)
            d_maxr = d_burner * (1.0 if t["clamped"] else 0.0)
            d_coil += d_needed + d_maxr * (1.0 if t["coil_heat"] > 0.0 else 0.0)
            d_loss += d_needed + d_maxr
            d_tank += -d_needed * Cw / dt
            d_hws_sp += d_needed * Cw / dt
            # coil_heat = w_cap*(tank-ret_new)
            d_wcap += d_coil * (t["tank"] - t["ret_new"])
            d_tank += d_coil * t["w_cap"]
            d_ret = -d_coil * t["w_cap"]
            # tank dissipation (linear in tank beyond the clamp)
            if t["tank"] >= t["amb1"]:
                r_int = self.h_radius
                r_ext = self.h_radius + self.h_insul_thick
                denom = (np.log(r_ext / r_int) / self.h_insul_cond
                         + 1.0 / self.h_conv / r_ext)
                d_tank += d_loss * self.h_len * 2.0 * np.pi / denom
            # return water
            if t["ret_active"]:
                ret_raw = t["ret_raw"]
                if ret_raw < t["lo"]:
                    d_ret_raw = 0.0
                    d_lo, d_hi = d_ret, 0.0
                elif ret_raw > t["hi"]:
                    d_ret_raw = 0.0
                    d_lo, d_hi = 0.0, d_ret
                else:
                    d_ret_raw = d_ret
                    d_lo = d_hi = 0.0
                air = t["air_temp"]
                if t["tank"] < air:
                    d_tank += d_lo
                    d_air = d_hi
                else:
                    d_air = d_lo
                    d_tank += d_hi
                if t["total_reheat"] >= 0.0:
                    d_coil_min += d_air
                else:
                    d_coil_max += d_air
                d_tank += d_ret_raw
                d_total_reheat += -d_ret_raw / t["w_cap"]
                d_wcap += d_ret_raw * t["total_reheat"] / t["w_cap"] ** 2
                dret_prev = np.zeros(1)
            else:
                dret_prev = np.array([dret_in + d_ret])
            d_total_flow += d_wcap * WATER_CP * WATER_DENSITY
        else:
            # shipped tree: the tank moves at the observation read; the gas is
            # billed from the setpoint, the return temperature and the step's
            # tank change.
            d_tank_new = dtank_in
            sp = t["act"][2 * self.nv + 3 * self.na]
            if sp > t["tank"]:
                heated = t["tank"] + self.h_heating_rate * dt / 60.0
                if heated < sp:
                    d_tank += d_tank_new
                else:
                    d_hws_sp += d_tank_new
            elif sp < t["tank"]:
                cooled = t["tank"] - self.h_cooling_rate * dt / 60.0
                if cooled > sp:
                    d_tank += d_tank_new
                else:
                    d_hws_sp += d_tank_new
            else:
                d_hws_sp += d_tank_new
            # burner
            db = d_gas
            supply = max(sp, t["ret_new"])
            flow_heat = WATER_CP * WATER_DENSITY * t["total_flow"] * (supply - t["ret_new"])
            dissip = tank_dissipation(supply, t["amb1"], self.h_len, self.h_radius,
                                      self.h_insul_thick, self.h_insul_cond, self.h_conv)
            tank_heat = Cw * (t["tank_new"] - t["tank"]) / dt
            total = flow_heat + dissip + tank_heat
            if total > 0.0 and db != 0.0:
                d_flow_heat = db
                d_total_flow += d_flow_heat * WATER_CP * WATER_DENSITY \
                    * (supply - t["ret_new"])
                d_ret = -d_flow_heat * WATER_CP * WATER_DENSITY * t["total_flow"]
                if sp > t["ret_new"]:
                    d_hws_sp += d_flow_heat * WATER_CP * WATER_DENSITY * t["total_flow"]
                else:
                    d_ret += d_flow_heat * WATER_CP * WATER_DENSITY * t["total_flow"]
                if supply >= t["amb1"]:
                    r_int = self.h_radius
                    r_ext = self.h_radius + self.h_insul_thick
                    denom = (np.log(r_ext / r_int) / self.h_insul_cond
                             + 1.0 / self.h_conv / r_ext)
                    d_diss = db * self.h_len * 2.0 * np.pi / denom
                    if sp > t["ret_new"]:
                        d_hws_sp += d_diss
                    else:
                        d_ret += d_diss
                # tank_heat = Cw*(tank_new - tank)/dt
                d_tank_new += db * Cw / dt
                d_tank -= db * Cw / dt
            # tank_new from tank_temperature: add the step_change chain
            dret_prev = np.zeros(1)
            d_tank += 0.0
            # (the tank_new gradient already flowed through d_tank_new above)
        return dict(d_total_flow=d_total_flow, d_head=d_head, d_tank=d_tank,
                    dret_prev=dret_prev, d_total_reheat=d_total_reheat,
                    d_hws_sp=d_hws_sp, d_coil_min=d_coil_min,
                    d_coil_max=d_coil_max, d_hws_sp_conv=d_hws_sp)
