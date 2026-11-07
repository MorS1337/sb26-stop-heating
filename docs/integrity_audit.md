# Can simulator artifacts explain the public leaderboard? A measured audit

Analysis-only study (2026-11-06). Nothing here is submitted: the probes that
exploit the artifacts live in `policies/exploit_b1.py` and
`policies/exploit_b2.py`, both of which say so in their docstring.

Two accounting artifacts of the shipped simulator are known and both have a
public fix (starter-kit PR #2, PR #3). This report measures, on the anchor
episode the forum uses for comparison (`building_train_1/fairbanks_1`, default
start, first 2016 steps = 7 days) and on the 10-scenario VoI subset:

1. **how much heat each artifact can manufacture**, in kWh, from instrumented
   rollouts that do not perturb the simulator (proof in section 2);
2. **how much reward that heat is worth** -- and it is not the same number,
   because manufactured heat is not automatically heat that displaces fuel;
3. **how much of it survives the corrected simulator** (PR #2 + PR #3), which is
   the honest measure of an artifact's worth: the difference between the same
   policy on the two trees;
4. whether the resulting headroom can plausibly explain the public anchor gap
   (our CtrlW −93.12, MOBI-EVERGi −81.9, i.e. 11.22 reward units).

Headline: **the boiler-billing artifact (B1) is worth +36.93 reward units
(+39.7 %) to CtrlW on the anchor at *zero* comfort cost, and vanishes on the
corrected tree. The return-air artifact (B2) is worth 0.7 % of CtrlW's coil heat
(182 kWh in a week) and is worth *nothing* in reward terms to any
comfort-respecting controller: CtrlW, Wu and zt-best all score slightly
*better* on the corrected tree.** B2 can be made large only by policies that
lose 58 to 10 009 reward units of comfort producing it.

---

## 1. Setup

| item | value |
|---|---|
| anchor | `building_train_1/fairbanks_1`, default start, 2016 steps (7 days), total reward |
| simulator trees | `sims/shipped` (as the LB runs it), `sims/corrected` (PR #2 energy-balanced tank + PR #3 airflow-weighted return air) |
| runner | `python -m sbx.integrity` (`sbx/integrity.py`), `--fast` (bit-identical to plain, 1.6-1.9x faster) |
| reward weights | productivity 0.6, energy 0.2, carbon 0.2; comfort deviation reference 5 K (`config.gin` 296-312) |
| reward decomposition | `total = -0.6 · Σ normalized_productivity_regret − 0.2 · Σ normalized_energy_cost − 0.2 · Σ normalized_carbon_emission` |

CtrlW on the anchor (`results/houssem_anchor_ctrlw`, plain `sbx.evalrun`,
bit-identical to the instrumented run):

| term | value |
|---|---|
| total reward | **−93.1229** |
| comfort (`-0.6·Σnpr`) | −0.541 |
| energy (`-0.2·Σec`) | −26.23 |
| carbon (`-0.2·Σce`) | −66.35 |
| gas billed | 27.19 MWh in 7 days |
| coil heat delivered | 26.72 MWh |

So the anchor is a **fuel bill**: 92.6 of the 93.1 units are gas burned to heat
a Fairbanks winter, and comfort is 0.6 % of it.

### Commands

```bash
VENV=starter-kit/.venv/bin/python

# instrumented anchor, one policy, one tree
$VENV -m sbx.integrity run --policy policies.ctrl_w:CtrlW \
    --scenario building_train_1/fairbanks_1 --steps 2016 --sim shipped \
    --fast --workers 1 --tag ia_ctrlw_shipped

# the Wu-equivalent (CtrlW with its five layers switched off)
$VENV -m sbx.integrity run --policy policies.ctrl_w:CtrlW \
    --params '{"precondition_hours": 1.0, "release_enabled": false,
               "reward_weights_enabled": false, "holiday_guard_enabled": false,
               "early_preheat_enabled": false}' \
    --scenario building_train_1/fairbanks_1 --steps 2016 --sim shipped \
    --fast --workers 1 --tag ia_wu_shipped

# zt-best and the organizer pid_water baseline: same command, --policy
#   policies.ref_ztbest:RefZtBest   /   baseline:pid_water

# B1 probe (hot-water setpoint sawtooth), both trees
$VENV -m sbx.integrity run --policy policies.exploit_b1:ExploitB1 \
    --scenario building_train_1/fairbanks_1 --steps 2016 --sim shipped \
    --fast --workers 1 --tag ia_b1_shipped        # and --sim corrected

# B1 probe + the recorded-schedule oracle (analysis only, --oracle)
$VENV -m sbx.integrity run --policy policies.exploit_b1:ExploitB1Foresight \
    --params '{"foresight_preheat_lead_hours": 5.0}' \
    --scenario building_train_1/fairbanks_1 --steps 2016 --sim shipped \
    --fast --oracle foresight \
    --oracle-file $PWD/scratch/VoIOracle/foresight_ref --tag ia_b1f_shipped
# NB --oracle-file must be absolute: simload chdirs into the sim tree.

# B2 probe, the four variants reported here
$VENV -m sbx.integrity run --policy policies.exploit_b2:ExploitB2 \
    --params '{"b2_blend": 1.0, "b2_ramp": "linear", "b2_keep_flow": true,
               "b2_valve_mode": "base"}'  --scenario ... --tag ia_v_lin_shipped
$VENV -m sbx.integrity run --policy policies.exploit_b2:ExploitB2 \
    --params '{"b2_ramp": "binary", "b2_pressure_native": 1.0,
               "b2_valve_mode": "flood_off"}' --scenario ... --tag ia_v_p2000_shipped
$VENV -m sbx.integrity run --policy policies.exploit_b1:ExploitB1B2 \
    --params '{...b2 linear...}' --scenario ... --tag ia_b1b2_shipped

# non-perturbation proof: three rollouts of the same episode
#   full hooks (default) | --reward-only | --no-hooks
$VENV -m sbx.integrity run --policy policies.ctrl_w:CtrlW --scenario ... \
    --steps 2016 --sim shipped --fast --reward-only --tag ia_proof_reward_shipped
$VENV -m sbx.integrity run --policy policies.ctrl_w:CtrlW --scenario ... \
    --steps 2016 --sim shipped --fast --no-hooks   --tag ia_proof_nohook_shipped
$VENV scratch/IntegrityAudit/proof.py

# tables
$VENV scratch/IntegrityAudit/report.py anchor    # the main table
$VENV scratch/IntegrityAudit/report.py probes    # artifact magnitudes
$VENV scratch/IntegrityAudit/report.py cooling   # cooling-direction screen
$VENV scratch/IntegrityAudit/report.py subset    # 10-scenario subset
```

The two campaign scripts are `scratch/IntegrityAudit/anchor_queue{,2,3}.sh`
(anchor, both trees, two simulator processes at a time) and
`scratch/IntegrityAudit/cooling_queue.sh`, `subset_queue2.sh`.

---

## 2. The B2 account: definition, sign, and a hook that does not perturb

### 2.1 The quantity

The shipped simulator gives each air handler the **floor-area** mean of the
temperatures of the zones its VAVs serve
(`simulator_flexible_floor_plan.py` ≈L318-330 for the step, ≈L743-755 for the
cooling bill), while a zone exchanges heat with the supply air at **its own
airflow**. For recirculation `r`, total mass flow `mdot` and the two candidate
return temperatures the imbalance is (Kozzen, topic 753804 / PR #3)

    dQ = r · mdot · c_p · (T_area_weighted − T_airflow_weighted)

`sbx/integrity.py` computes exactly this per air handler, per step, plus:

* `supply_gap` — the part that reaches the zones. The handler's supply-air
  temperature is built from the area-weighted return, so with the coil
  unclamped the air handed to every zone is `r·ΔT` warmer than a consistent
  simulator would give; times the delivered flow that is `mdot·c_p·r·ΔT`, i.e.
  the same number whenever the mixed air is below the supply setpoint. The two
  columns agree to the last digit on every shipped run in this report
  (182 vs 182 kWh for CtrlW), which is the check that this really is heat
  reaching the zones and not a bookkeeping residue.
* the mixing-box entropy of the streams that really exist (recirculated extract
  at each VAV's own zone temperature, plus outdoor air) ending at the mixed
  temperature the simulator uses. Negative is thermodynamically impossible.
* the **same-state ceiling**: `r · mdot_rated · c_p · (T_area − T_coldest zone)`,
  i.e. the most favourable split the *current* temperature field allows. It is a
  bound on the artifact at a fixed trajectory, not an achievable number.

### 2.2 The sign, derived

`dQ > 0` (manufactured heat) needs `T_area > T_flow`, i.e. the airflow-weighted
return **colder** than the area mean: send the air through the **coldest zones**
and hold the warmest at the ventilation floor. The reverse pattern manufactures
free cooling. Both directions are implemented (`b2_sign`), and the sign is not a
free choice: in a heating-dominated episode `dQ > 0` needs air in the coldest
zones, which is also where air *cools* the building hardest — that coupling is
the whole story of section 3.

### 2.3 The hook, and the proof that it is transparent

The sample is taken **inside** `execute_step_sim`, at the moment the handler
computes its supply-air temperature: `vav.update_settings` has set the dampers,
the thermal solve has not run, so both the return temperature the simulator uses
and the flows the VAVs asked for are simultaneously current. Nothing is
re-read from a sensor that has side effects -- in particular the boiler's
`supply_water_temperature_sensor`, which advances the tank in the shipped tree,
is never touched: the burned gas is captured by wrapping
`boiler.compute_thermal_energy_rate` (shipped) / `apply_heat_draw` (corrected),
both of which return their original value unchanged.

Three rollouts of the same episode, differing only in what is installed
(`--no-hooks` installs nothing, `--reward-only` installs only the reward wrapper
so the reward's own components can be compared):

| quantity | hook vs reward-only | reward-only vs nothing | max abs difference |
|---|---|---|---|
| per-step `reward` | bit-identical | bit-identical | 0.000e+00 |
| per-step `npr` | bit-identical | not logged | 7.941e-02 |
| per-step `ec` | bit-identical | not logged | 3.615e-01 |
| per-step `ce` | bit-identical | not logged | 8.516e-01 |
| per-step `occupancy` | bit-identical | not logged | 7.400e+01 |

totals: full −93.12290938248043, reward-only −93.12290938248043, nothing
−93.12290938248043; bit-identical across all three. The plain `sbx.evalrun`
record of the same episode (`results/houssem_anchor_ctrlw`) is
−93.12290938248043, i.e. the whole instrumented run reproduces the production
record to the last bit.

On the corrected tree the same table holds
(−92.78747206777916 in all three modes).

### 2.4 An independent validation: the corrected tree must read zero

`simg/corrected` computes the return temperature with the airflow-weighted
mixture, so this instrument -- which recomputes the mixture itself -- must
report exactly zero imbalance there. It does, on every policy measured:

| policy | mean imbalance, shipped | mean imbalance, corrected |
|---|---|---|
| CtrlW | +1.08 kW | +0.00 kW (max abs 2.2e-13 W) |
| Wu | +1.09 kW | +0.00 kW |
| zt-best | +1.77 kW | −0.00 kW |
| pid_water | +10.23 kW | +0.00 kW |
| B2 probe (linear) | +13.99 kW | −0.00 kW |
| B2 probe (2000 Pa) | +242.91 kW | +0.00 kW |

The residual is float rounding, so the formula reproduces PR #3's weighting
exactly rather than approximately. (This is the internal-consistency check, not
a claim that the corrected tree is artifact-free: it also removes B1.)

---

## 3. B2 measured: what it is worth to real controllers, and to a probe

### 3.1 The main table

Anchor, 2016 steps, both trees. `free heat` is `Σ max(dQ,0)·Δt` in kWh;
`of coil heat` is that over the VAV reheat the coils delivered in the same
episode; `artifact worth` is *shipped total − corrected total* for the same
controller, i.e. what the artifacts are worth to that trajectory.

| controller | sim | anchor total | Δ vs CtrlW | Δ % | comfort | energy | carbon | free heat kWh | of coil heat % | gas billed kWh | artifact worth |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CtrlW | shipped | -93.123 | +0.000 | +0.0 | -0.541 | -26.2 | -66.3 | 182 | 0.7 | 27187 | -0.34 |
|  | corrected | -92.787 | +0.000 | +0.0 | -0.551 | -26.1 | -66.1 | 0 | 0.0 | 27087 |  |
| Wu (CtrlW, five flags off) | shipped | -95.003 | -1.880 | -2.0 | -1.959 | -26.4 | -66.7 | 183 | 0.7 | 27334 | -0.42 |
|  | corrected | -94.581 | -1.794 | -1.9 | -1.968 | -26.3 | -66.3 | 0 | 0.0 | 27210 |  |
| zt-best | shipped | -105.173 | -12.050 | -12.9 | -0.707 | -30.7 | -73.8 | 303 | 1.0 | 30053 | -0.86 |
|  | corrected | -104.317 | -11.530 | -12.4 | -0.734 | -30.7 | -72.9 | 0 | 0.0 | 29643 |  |
| pid_water (organizer) | shipped | -596.865 | -503.742 | -540.9 | -189.96 | -157.2 | -249.7 | 1724 | 1.7 | 95764 | +166.53 |
|  | corrected | -763.390 | -670.603 | -722.7 | -231.34 | -206.8 | -325.3 | 0 | 0.0 | 124525 |  |
| B1 probe (sawtooth) | shipped | **-56.189** | **+36.934** | **+39.7** | -0.539 | -16.3 | -39.4 | 182 | 0.7 | 16055 | +37.19 |
|  | corrected | -93.380 | -0.592 | -0.6 | -1.360 | -26.2 | -65.9 | 0 | 0.0 | 26980 |  |
| B1 probe + occupancy foresight | shipped | **-55.510** | **+37.613** | **+40.4** | -0.138 | -16.3 | -39.1 | 107 | 0.4 | 15910 |  |
| B2 probe (linear ramp, constant flow) | shipped | -243.341 | -150.218 | -161.3 | -58.15 | -57.8 | -127.4 | 2370 | 4.6 | 51382 | +12.50 |
|  | corrected | -255.838 | -163.051 | -175.7 | -61.17 | -60.8 | -133.8 | 0 | 0.0 | 53945 |  |
| B2 probe (max pressure, flooded coils off) | shipped | -4707.708 | -4614.585 | -4955 | -4494.5 | -131.3 | -81.9 | 40810 | 183.8 | 22370 | +5550.36 |
|  | corrected | -10258.068 | -10165.280 | -10955 | -10009.2 | -141.5 | -107.4 | 0 | 0.0 | 32820 |  |
| B1+B2 probe | shipped | -157.000 | -63.877 | -68.6 | -58.73 | -34.4 | -63.9 | 2370 | 4.6 | 25153 | +89.50 |
|  | corrected | -246.502 | -153.714 | -165.7 | -63.88 | -56.3 | -126.4 | 0 | 0.0 | 51065 |  |

Read the `artifact worth` column as the answer to "what would this trajectory
gain if the accounting were fixed": it is the *only* clean measure of an
artifact's value, because it holds the policy fixed and changes nothing but the
simulator.

### 3.2 What the table says

**B2 is essentially absent for the controllers that are graded well.** CtrlW
manufactures 182 kWh in a week -- 0.7 % of the 26.7 MWh of coil heat it buys --
and its artifact worth is **−0.34**: the corrected tree is *cheaper* for it, so
the artifact is not a gain at all. Wu and zt-best are the same (−0.42, −0.86).
An honest controller moves little air on a Fairbanks week (CtrlW 4.8 m³/s mean
against a rated 80 m³/s for the four handlers), and the artifact is proportional
to the air it moves.

**B2 is present for controllers that move a lot of air badly.** pid_water runs
29 m³/s mean and gets 1 724 kWh of free heat (1.7 % of its coil heat) -- and the
corrected tree then costs it **+166.5** units, 28 % of its own score. This is
the mechanism Kozzen describes ("+42.75 kW for the PID baseline"), and it is a
*penalty on the corrected tree* rather than a bonus on the shipped one: the PID
baseline is bad for other reasons, and the artifact merely hides 28 % of its
bill.

**The artifact's size is bounded by the fan, and the fan is priced.** The two
fans cost `flow·P/0.9` (intake) plus `flow·(1−r)·P/0.9` (exhaust), i.e.
`1.44·flow·P` W at r = 0.7, while the artifact pays `r·ρ·c_p·flow·ΔT` =
`845·flow·ΔT` W of heat that is worth 0.48 of a watt of electricity (the
marginal gas/electric ratio derived in `policies/ctrl_w.py`). Both scale with
flow, so flow cancels and the trade clears only above

    ΔT_break-even ≈ P / 282 K     (1.8 K at 500 Pa, 3.6 K at 1 kPa, 7.1 K at 2 kPa)

Measured against that line: CtrlW's own field supplies **0.12-0.59 K** of
area-vs-airflow difference, the linear probe reached **1.2 K** at constant flow,
the 1 kPa stress probe **6.2 K** (163 kW at 31 m³/s — just under its
break-even), and the 2 kPa probe **6.6 K** at 44 m³/s, where its 243 kW of
manufactured heat is paid for by 126 kW of fans. And the air has to be
*concentrated* to make `T_area ≠ T_flow` at all: opening every damper evenly
gives `T_area = T_flow` and `dQ = 0` at any flow, so the flow that manufactures
heat is exactly the flow that must be starved somewhere else.

**Manufacturing heat costs the comfort term far more than it saves.** The B2
linear probe multiplies the free heat by 13x (182 → 2 370 kWh, 4.6 % of its coil
heat) and its artifact worth is only 12.5 units, because redistribution cools
the flooded zones: the probe's *shipped* score is 150 units worse than CtrlW and
its energy+carbon terms alone are 3.7x CtrlW's. The aggressive variant
manufactures 40.8 MWh — more than the whole gas bill, 183 % of the coil heat it
delivers — and buys 5 550 units of artifact worth at the cost of 4 494 units of
comfort on the shipped tree and 10 009 on the corrected one. `pid`-class
policies sit in between. With `productivity_weight = 0.6` and an unbounded
shortfall beyond 5 K, the reward prices a temperature-spread strategy at 3x the
energy weight, so this trade is not survivable: the artifact can only be
harvested by a policy that has given up on comfort.

**The same-state ceiling shows why nobody bothers.** At CtrlW's own temperature
field the most favourable split allowed by the rated flows would manufacture
24.5 MWh — the size of the entire gas bill. Realising it would need the whole
rated flow of each handler through its coldest zone, i.e. a building at a
permanent 3-6 K spread, which the comfort term prices at hundreds of units per
day. The gap between 182 kWh and 24 463 kWh is the distance between a real
controller and a stress test.

### 3.3 The cooling direction, measured separately

In cooling mode the handler clamps at its supply setpoint, so the delivered air
does not change and only the *bill* moves; the flow direction that manufactures
free cooling is airflow to the **warmest** zones — the direction comfort likes.
That makes it the one configuration where B2 could in principle be nearly free.
Four summer scenarios, 2016 steps, shipped tree:

| controller | scenario | mean reward ×100 | Δ score ×100 | comfort | energy | carbon | free cooling kWh |
|---|---|---|---|---|---|---|---|
| CtrlW | phoenix_5 | -1.219 | +0.000 | 0.000 | 99.8 | 23.0 | 6 |
| B2 cooling ramp (constant flow) | phoenix_5 | -2.377 | +1.159 | -18.417 | 146.4 | 38.0 | 1005 |
| B2 cooling, binary + 1000 Pa | phoenix_5 | -7.302 | +6.083 | -57.425 | 444.5 | 119.3 | 3042 |

The manufactured cooling is real (1.0 and 3.0 MWh) and the entropy test fires
(94 % of steps have negative mixing-box entropy, versus 55 % for CtrlW itself,
versus 0 % on the heating side, where the artifact is an energy surplus rather
than an entropy violation). But the probes still lose by 1.2 and 6.1 score
points: tripling the airflow to concentrate it costs more fan energy than the
under-billed chiller load, and starving the cool half overheats it. CtrlW itself
already sits at 6 kWh of free cooling in this window — it is not leaving money
on the table. (The first 7 days of this scenario are a mild Phoenix February;
the 30-day subset in section 5 is the wider test.)

---

## 4. The B1 account: the sawtooth

The shipped boiler bills

    gas = max(0, c_w·rho_w·flow·(setpoint − return_water) + standing loss + tank step change)

while the coils draw at the tank's *actual* temperature, and the tank falls at a
fixed `cooling_rate` (0.1 K/min) when the setpoint drops. `policies/exploit_b1.py`
applies the published recipe (setpoint max below 345 K, min at or above it) on
top of CtrlW's own action, changing nothing else.

| | CtrlW shipped | B1 probe shipped | B1 probe corrected |
|---|---|---|---|
| anchor total | −93.123 | **−56.189** (+36.93, +39.7 %) | −93.380 |
| comfort | −0.541 | −0.539 | −1.360 |
| energy | −26.23 | −16.28 | −26.15 |
| carbon | −66.35 | −39.37 | −65.87 |
| gas billed | 27.19 MWh | **16.06 MWh (−40.9 %)** | 26.98 MWh |
| coil heat delivered | 26.72 MWh | 26.72 MWh | 26.87 MWh |
| billed / delivered | 1.018 | **0.601** | 1.004 |

That is the artifact in one line: the same 26.7 MWh of reheat is delivered to the
building while 11.1 MWh of gas stops being billed, and **comfort does not move**
(−0.539 against −0.541). On the corrected tree the arbitrage is gone
(billed/delivered 1.004) and the same policy is 0.59 units *worse* than CtrlW,
with comfort more than twice as bad, because a tank that actually cools delivers
colder water.

B1 is independent of B2 and additive: the B1 sweep leaves the B2 imbalance
untouched (182 kWh, 0.7 %), and B1+B2 on the anchor is −157.00, i.e. the B2
probe's own −150 units of damage dominate the B1 gain.

With the recorded-schedule oracle on top (`policies/exploit_b1:ExploitB1Foresight`,
5 h lead, analysis-only channel) the anchor reaches **−55.510 (+37.61, +40.4 %)**,
with comfort improving to −0.138 -- but most of that comfort gain is the foresight
policy, and the value of the foresight *information* collapses when fuel is
nearly free: it added +2.60 units to CtrlW (`docs/voi_oracle.md` §9) and
+0.68 on top of B1.

---

## 5. Beyond the anchor: the subset, and what is still running

The subset is the 10 scenarios of `docs/voi_oracle.md` (2 per station, 30-85
zones). One full-length (8641-step) pair finished before this audit closed, for
CtrlW on both trees; it is the only subset number here that is complete, and it
is reported alone rather than extrapolated.

`building_train_1/atlanta_2`, 8641 steps, CtrlW:

| sim | total reward | gas billed kWh | comfort (`Σnpr`) | free heat kWh |
|---|---|---|---|---|
| shipped | −27.73117 | 5403 | 0.931 | 32.3 |
| corrected | **−24.84054** | 4493 | 0.991 | 0 |

The corrected tree is **2.89 units better (10.4 %)** for CtrlW on Atlanta, and
the shipped tree makes it burn **20 % more gas** (5403 against 4493 kWh). The
artifact is not a one-way gift: on this scenario it *hurts* the controller,
because the area-weighted return biases the mixed air the wrong way for this
weather. Together with the anchor (shipped 0.34 units worse for CtrlW) the two
complete scenarios say the same thing from the other side: **our own controller
loses a little to the shipped accounting; it does not gain from it.**

The 10-scenario campaign at both step counts is scripted and reproducible but
was not finished inside this audit:
`bash scratch/IntegrityAudit/subset_queue2.sh ctrlw b1 b2` (per-scenario
processes, shipped and corrected side by side, `STEPS` selects 2016 or 8641;
`scratch/IntegrityAudit/report.py subset` prints the table from
`results/is_*`). Two more points did land, both on `atlanta_2`:

| controller | steps | sim | total | gas billed kWh |
|---|---|---|---|---|
| CtrlW | 2016 | shipped | −2.5603 | 405 |
| CtrlW | 2016 | corrected | **−1.6402** | 112 |
| B1 probe | 8641 | shipped | −35.5433 | 7777 |
| B1 probe | 8641 | corrected | **−24.9710** | 4534 |

Two things follow, and both cut against the exploit reading. First, CtrlW is
**36 % better** on the corrected tree in the anchor's own 7-day window on
Atlanta: the shipped accounting is a *handicap* for our controller on mild
scenarios, not a gift. Second, the B1 probe is **7.8 units worse than CtrlW** on
Atlanta while being 36.9 units better on the anchor: the sawtooth as published
is ungated, so on a mild scenario it heats a tank nothing is drawing from (7.78
against 5.40 MWh of gas for CtrlW on the same tree). That is a defect of this
probe, not a property of the artifact -- gating the recipe on the same
`recent_heat` predicate `CtrlW` already uses is the obvious next run -- but the
honest statement of what is measured is that **B1's worth is
scenario-dependent and can be negative for an untuned sawtooth**.
**[INFERENCE]** Its subset average is therefore materially smaller than the
anchor's +39.7 %; the anchor figure is the ceiling, reached on the most
heating-dominated station, and should be quoted as a ceiling rather than as a
leaderboard-wide number.

---

## 6. Verdict: can the artifacts explain −81.9 on the anchor?

**Measured, the headroom exists but the sign is wrong for an honest controller,
and the one channel that pays is the one MOBI says they do not use.** B1 is
worth **+36.93 reward units (+39.7 %) to CtrlW** on the anchor -- three times the
11.22-unit gap to MOBI's −81.9 -- at zero comfort cost and with 40.9 % of the gas
bill disappearing, but it is worth **+0.59 units of *loss*** on the corrected
tree, i.e. it is pure accounting. B2 is worth **−0.34 units to CtrlW** (−0.42 Wu,
−0.86 zt-best): an honest controller moves little air, the artifact is
proportional to air moved, and the shipped tree's area-weighted return happens to
be *adverse* for it -- the corrected tree is cheaper. B2 becomes large only as a
penalty on bad trajectories (+166.5 for pid_water, +5 550 for a probe that
abandons comfort entirely), and manufacturing the heat costs 58-10 009 units of
comfort because the airflow that creates the mismatch is the airflow that cools
the flooded zones, while the fan pays `1.44·flow·pressure` watts for the
privilege (break-even mismatch `ΔT ≈ P/282 K`, i.e. 7.1 K at full pressure,
against 0.1-0.6 K that CtrlW's own field supplies). So on the same episode: a
policy *could* have taken 37 units from the boiler billing mismatch, MOBI's own
PR #2 test says their controller took **0.3** (−81.9 → −81.6), and no
comfort-respecting controller can take more than a fraction of a unit from the
return-air weighting. Therefore **[INFERENCE]** MOBI's −81.9 is controller
quality, not artifact harvest -- a 12.1 % fuel reduction that our measurements
say must come from the plant and the schedule, not from the accounting. The
caveat that keeps this an inference rather than a proof: the shipped
leaderboard's headroom is *ample* for the other direction (B1 alone is 3.3x the
anchor gap, and a 55 % fuel cut -- the 0.00470 frontier against our 0.01051 --
fits inside B1's 41 % plus a B2-class penalty), nobody has denied exploiting
(forum 755748, zero replies), and a team that cycles the water setpoint is
invisible on any scoreboard until the simulator is re-run. The audit's
actionable statements are the two facts that need no inference: the two fixes
change an honest controller by *at most a few tenths of a unit* (and can change
it in its favour), and they remove from a dishonest one 37 units on the boiler
and up to 166 on the return air.

---

## 7. Files and provenance

| tag pattern | what |
|---|---|
| `results/ia_ctrlw_{shipped,corrected}` | CtrlW, anchor, instrumented |
| `results/ia_wu_*`, `ia_ztbest_*`, `ia_pidw_*` | the other three base controllers |
| `results/ia_b1_*` | B1 sawtooth probe, both trees |
| `results/ia_b1f_shipped` | B1 + recorded-schedule oracle |
| `results/ia_v_{lin,bin,p1000,p2000}_*` | B2 probe variants |
| `results/ia_b1b2_*` | B1 + B2 |
| `results/ia_proof_{hooked,reward,nohook}_{shipped,corrected}` | non-perturbation proof |
| `results/ic_shipped_{ctrlw,coola,coolb}` | cooling-direction screen, 4 summer scenarios |
| `results/is_{ctrlw,b1,b2}_{shipped,corrected}` | 10-scenario subset, 8641 steps |
| `sbx/integrity.py` | the instrumented runner (`run`, `--reward-only`, `--no-hooks`) |
| `policies/exploit_b1.py` | B1 probe (+ B1+B2, + B1+foresight); analysis only |
| `policies/exploit_b2.py` | B2 probe (+ B2+foresight); analysis only |
| `scratch/IntegrityAudit/proof.py` | the three-way bit-identity check |
| `scratch/IntegrityAudit/report.py` | every table in this report |
| `scratch/IntegrityAudit/{anchor_queue,anchor_queue2,anchor_queue3,cooling_queue,subset_queue2}.sh` | campaign scripts |
