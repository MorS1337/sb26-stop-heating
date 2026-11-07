# Adaptive preheat (W1b) — CtrlW2 final report

> **Follow-up task done: CtrlW4 (night lift on CtrlW2) — verdict GO for dev30.**
> See the last section, "CtrlW4: the fixed 2 K pre-dawn night lift stacked on
> CtrlW2".  Everything before it is the CtrlW2/W1b study.

**Verdict: CtrlW2 beats CtrlW, but by 0.16%, and it is a *small* effect that
trades the paper's hour-06 comfort metric for dawn energy.**  Full dev30
(shipped, 8641 steps, all 30 scenarios): **CtrlW2 0.0073575 vs CtrlW 0.0073694
(−0.161%)**, better on 14/30 scenarios, no scenario worse by more than 0.66%,
none worse by >3%.  Recommended variant **v4 = trim+extend**, which is now the
`CtrlW2.PARAMS` default:

```
preheat_min_hours 1.0   preheat_max_hours 3.0   preheat_gate_k 2.0
preheat_safety    1.0   preheat_slack_hours 0.25  preheat_fallback_hours 2.0
```

Run: `--policy policies.ctrl_w2:CtrlW2` with no `--params` (or the JSON above).
`preheat_adaptive_enabled=false` is bit-exact CtrlW (verified: 400 steps of
train_1/fairbanks_1, total −4.951023667475965 for both).

## What was measured

Everything below is full 8641-step episodes, `sims/shipped`, `--components hook`,
scored the leaderboard way (mean over scenarios of |mean per-step reward|, lower
is better).  Only the mean score is sensitive enough to see these effects; hour-06
comfort is the paper's arrival metric (occupancy-weighted comfort sum in the
06:00 hour bucket), 03-06 energy is the 03:00-06:59 energy sum.

### A. 12-scenario subset (6 fairbanks + 6 phoenix, the cold/warm extremes)

| variant | params (min/max/gate/safety/slack) | score | Δ vs CtrlW | comfort | energy | carbon | better | worst scen. |
|---|---|---|---|---|---|---|---|---|
| CtrlW (`results/ctrw_all`) | — | 0.011201 | — | −6.21 | −502.86 | −652.36 | — | — |
| v2 extend-only (`ctw2_v2`) | 2.0/3.0/2.0/1.0/0.25 | 0.011196 | **−0.04%** | −5.75 (**+7.4%**) | −502.73 (+0.0%) | −652.48 (−0.0%) | 8/12 | +0.07% |
| trim-only (`ctw2_trim`) | 1.0/2.0/1.0/1.25/0.25 | 0.011182 | **−0.16%** | −7.05 (−13.5%) | −502.05 (+0.2%) | −650.42 (+0.3%) | 8/12 | +0.00% |
| **v4 trim+extend** (`ctw2_v4`) | 1.0/3.0/2.0/1.0/0.25 | **0.011180** | **−0.18%** | −6.64 (−6.9%) | −501.96 (+0.2%) | −650.69 (+0.3%) | 8/12 | +0.37% |

Per scenario (score / Δ% / hour-06 comfort for v4 / 03-06 energy change):

| scenario | CtrlW | v2 | trim | **v4** | h06 CtrlW→v4 | Δ03-06en |
|---|---|---|---|---|---|---|
| train_1/fairbanks_1 | 0.045730 | −0.09 | −0.08 | **−0.21** | −2.189 → −2.323 | −0.14 |
| train_2/fairbanks_2 | 0.012412 | −0.15 | −0.33 | **−0.46** | −0.514 → −0.612 | +0.15 |
| train_4/fairbanks_4 | 0.001606 | +0.03 | −0.48 | **−0.28** | −0.289 → −0.299 | +0.02 |
| train_5/fairbanks_5 | 0.001553 | −0.00 | +0.00 | **+0.00** | −0.086 → −0.086 | +0.01 |
| train_7/fairbanks_7 | 0.008450 | +0.07 | −0.69 | **−0.36** | −0.469 → −0.583 | +0.14 |
| train_8/fairbanks_8 | 0.024166 | −0.01 | −0.33 | **−0.26** | −0.336 → −0.428 | +0.32 |
| train_1/phoenix_5 | 0.014240 | +0.00 | +0.00 | **+0.00** | 0.000 → 0.000 | 0.00 |
| train_2/phoenix_6 | 0.007199 | −0.00 | −0.00 | **−0.00** | −0.000 → −0.000 | 0.00 |
| train_4/phoenix_8 | 0.001644 | −0.01 | +0.00 | **+0.37** | −0.221 → −0.235 | +0.02 |
| train_5/phoenix_1 | 0.002350 | −0.00 | −0.00 | **−0.26** | −0.891 → −0.946 | +0.06 |
| train_7/phoenix_3 | 0.005454 | +0.00 | +0.00 | **+0.00** | −0.000 → −0.000 | 0.00 |
| train_8/phoenix_4 | 0.009606 | −0.00 | −0.00 | **−0.00** | 0.000 → 0.000 | 0.00 |

### B. Full dev30, v4 (`--set dev30`, 30 scenarios)

**CtrlW2 v4 = 0.007357548 vs CtrlW 0.007369419 → −0.161%**; better on 14/30,
worse on 8 (worst `train_1/denver_7 +0.66%`), identical (to 6 dp) on 8.
By term over 30 scenarios: comfort −15.40 (−7.15%), energy −966.75 (+0.14%),
carbon −925.15 (+0.30%), hour-06 comfort −9.987 → −11.057 (−1.07).

| station (6 scenarios each) | CtrlW | v4 | Δ |
|---|---|---|---|
| fairbanks | 0.015653 | 0.015612 | −0.26% |
| phoenix | 0.006749 | 0.006749 | +0.00% |
| singapore | 0.005708 | 0.005708 | +0.00% |
| atlanta | 0.004810 | 0.004799 | −0.23% |
| denver | 0.003927 | 0.003920 | −0.18% |

Fairbanks carries 42.5% of the dev30 sum and phoenix 18.3%, so the dev30 gain is
mostly the fairbanks morning; warm stations are untouched (their zones never bind
on the ramp band) and denver/atlanta move only where the identified response
differs from the frozen prior.

## Mechanism (why the direction was wrong at first)

CtrlW2 fits the *same* model Wu uses (`dT = d + g·x`, `x` = zone kW from
conductance·(supply−T) + reheat) per zone by RLS, then each night runs it forward
under full conditioning to predict the lead time each zone needs, and sets the
per-zone preheat horizon to `clip(safety·lead + slack, min, max)`, keeping the
fixed 2 h as fallback until identified and only exceeding it where the fallback
leaves the zone more than `gate_k` K short.  The ramp curve is Wu's own blend
generalised so that H = 2 h reproduces CtrlW exactly and the last hour is always
Wu's own 1 h ramp.

The static sweep (1 h / 2 h / 3 h) suggested the optimum was ≈2 h and that going
further bought comfort the reward would not pay for — so an adaptive horizon can
only redistribute, not shorten.  The measurements say otherwise:

* **Extending pays exactly as predicted, and only on cold zones** (v2: +7.4% of
  the remaining arrival comfort for +0.0% energy — the plant substitutes dawn
  heat for later heat nearly unit for unit).  But it is worth only −0.04%.
* **Shortening is the lever that pays**: v2 and trim both beat CtrlW, and trim
  beats v2 (−0.16% vs −0.04%).  In fairbanks all five scenarios improve
  (−0.08 … −0.69%) because the zones whose predicted lead is under 2 h no longer
  sit for two hours in a ramp band the identified model says they cannot follow:
  hour-06 comfort is 0.79 worse over the 12, but 03-06 energy is 2.1 cheaper and
  the reward (0.6 comfort / 0.2 energy / 0.2 carbon) prefers the trade.
* **v4 does both** and keeps ~all of trim's energy/carbon gain for half of trim's
  comfort loss.  v4 vs trim is a 0.02% tie in score at a strictly better
  comfort/energy trade, so v4 is the one shipped as the default.

Earlier claim now corrected: the note in the first version of
`policies/ctrl_w2.py` said trimming "removes the whole early ramp" in phoenix and
costs 0.2-3.7%.  It does not — phoenix is bit-identical to CtrlW under both trim
and v4 (see table A).  That 0.2-3.7% came from the *fixed* 1 h horizon, which
also flattens the ramp curve; with the generalised blend a 1 h horizon only skips
the pre-ramp, Wu's own ramp still runs, and the warm zones are above the band
either way.  The docstring in `policies/ctrl_w2.py` carries the corrected text.

Not promoted: `preheat_comp_gain = 1.0, preheat_comp_max = 1.5` (raise the band
by the measured ~1.5 K tracking lag) — 700 steps of fairbanks_1 gave
−14.9735 → −14.9832 (−0.07%) with hour-06 comfort unchanged (−0.0378 → −0.0376):
offsetting the band does not close the plant's proportional offset.  Default 0.0.
(`results/ctw2_comp_probe`, `ctw2_comp_probe2` are those 700-step probes and are
not comparable with the 8641-step rows.)

## Caveats

* The effect is 0.16% of the dev30 score.  It is real (deterministic per
  scenario, ~450× the measured machine-to-machine noise) but it will not move an
  LB rank on its own; its value for the paper is the diagnosis and the mechanism,
  not the digit.
* **The paper's hour-06 comfort metric regresses** (dev30 −9.99 → −11.06) while
  the reward improves.  That is the honest finding: on these buildings the arrival
  hour is over-preheated, and our own objective (and the competition's) prices the
  dawn energy above the last 0.5 K of comfort.  If the paper wants a *comfort*
  headline, v2 (extend-only, +7.4% hour-06 comfort, −0.04% score) is the right
  operating point and is one `--params` away.
* Reproducibility: identical params on one machine are bit-identical
  (`ctrw_a` == `ctrw_c` on fairbanks_1, total reward to the last bit).  The same
  run on the Mac (arm64) and on the PC (x86 WSL) differs by 3.6e-6 relative in
  total reward (`ctw2_trim` vs `ctw2_trim_check`; all 12 mean scores agree to
  6 dp) — three orders below the deltas above, so the two-machine mix in the dev30
  number below is immaterial.
* **The dev30 number is a two-machine set**: the 12 fairbanks/phoenix records come
  from `results/ctw2_v4` (PC, `--workers 6`), the 18 atlanta/denver/singapore
  records from `results/ctw2_v4_dev30` (Mac, `--workers 2`, `--fast`, which
  FastExact2 verified bit-identical to an unflagged run).  The full PC dev30 run
  (`/home/mors1337/sb26/results/ctw2_v4_dev30`, 6 workers) was launched before the
  PC went unresponsive (SSH banner timeouts — WSL memory pressure, not this job)
  and could not be pulled; it may still be running/finished there and is worth
  pulling when the PC answers, purely as a cross-check of the same 30-scenario set.

## CtrlW4: the fixed 2 K pre-dawn night lift stacked on CtrlW2

**GO — run the full dev30 (shipped + corrected) and take it to submission if that
confirms.**  `policies/ctrl_w4.py:CtrlW4(CtrlW2)` adds the *legal* half of the
VoI study (`docs/voi_oracle.md` §4-6): the lift that the forecast variant
switches on needs no forecast at all, and it stacks almost additively with the
adaptive horizon.  On the 18 scenarios measured (the 12 fairbanks+phoenix plus 6
more that cover all five stations) CtrlW4 is **−0.30% vs CtrlW2 v4 and −0.50% vs
CtrlW**, no scenario worse than +1.37%, with an estimated dev30 of
**0.0073415 (−0.22% vs v4, −0.38% vs CtrlW)** — three times the effect of the
adaptive horizon alone.

Command (no `--params`: the defaults are the measured configuration):

```
starter-kit/.venv/bin/python -m sbx.evalrun --policy policies.ctrl_w4:CtrlW4 \
    --set dev30 --steps 8641 --sim shipped --workers 6 --components hook \
    --tag ctw4_dev30_shipped
# and, for the paper:  --sim corrected --tag ctw4_dev30_corrected
```

`CtrlW4.PARAMS` = CtrlW2 v4 defaults + `night_lift_enabled true`,
`night_lift_K 2.0`, `night_lift_share 1.0`, `night_lift_lead_hours 4.0`,
`night_lift_floor_K 1.0`.  The module imports only `ctrl_w2` (→ `ctrl_w`
→ `wu_core`), reads no oracle/building id/weather file/occupancy seed, and adds
one vectorised `np.where` per band call (per-step cost unchanged within noise:
150-470 s per 8641-step episode, i.e. 17-54 ms/step with `--fast`).

### Mechanism

When CtrlW2's band call is in night mode (`now == False`), the zone floor `low`
is raised by `night_lift_K * night_lift_share` for every zone that satisfies all
of: `0 < hours_until(start_hour, hour) <= night_lift_lead_hours` (02:00-06:00 for
a 06:00 start), `_comfort_day(obs, False)` (CtrlW's own weekday/holiday
predicate), and `low <= standby_minimum + night_lift_floor_K` (the zone is
actually sitting on the standby floor).  The target is capped at
`comfort_band[0] - comfort_margin` and passed through CtrlW's one-step recovery
clamp.  This is a literal port of `OracleFcstCtrlW._night_lift` in its
no-oracle `fcst_night_fixed_share` mode; the rules and the arithmetic order are
unchanged, so the numbers reproduce.

### Exactness

2 scenarios × 576 steps, shipped, `--components hook`, `--fast`
(`results/w4chk_*`):

| scenario | CtrlW2 | CtrlW4 lift off | CtrlW4 lift on | StackLift (oracle fn on CtrlW2) |
|---|---|---|---|---|
| train_1/fairbanks_1 | −6.701032026147004 | **−6.701032026147004** | −6.449258128173824 | **−6.449258128173824** |
| train_8/singapore_7 | −2.8305741771855537 | **−2.8305741771855537** | −2.8305741771855537 | **−2.8305741771855537** |

* lift off == CtrlW2 **bit-identically**, and lift on == StackLift
  **bit-identically**, for `total_reward`, `components` and `by_hour`.
* `StackLift` (`scratch/AdaptivePreheat/stack_lift.py`) binds the unmodified
  `OracleFcstCtrlW._night_lift` onto CtrlW2 through the same call
  `OracleFcstCtrlW._band` makes, with `fcst_night_fixed_share = 1.0` — so the
  comparison checks the port *and* the wiring against the reference
  implementation, not against a copy of itself.
* Singapore_7 is unchanged with the lift on: the at-floor gate makes the
  mechanism inert in the cooling-dominated stations, as the VoI study reported.

### Results, 18 scenarios (8641 steps, shipped, Mac, `--fast`)

`results/ctw4_sub18` against `results/ctrw_all` and
`results/ctw2_v4`+`ctw2_v4_dev30`.  Δ% is versus CtrlW unless the column says
otherwise.

| scenario | CtrlW | v4 Δ% | **CtrlW4 Δ%** | CtrlW4 vs v4 | h06 CtrlW→CtrlW4 | Δ03-06en |
|---|---|---|---|---|---|---|
| train_1/atlanta_2 | 0.003209 | −0.22 | +0.69 | +0.91 | −0.419 → −0.630 | +0.04 |
| train_7/atlanta_8 | 0.007046 | −0.56 | **−1.40** | −0.85 | −0.435 → −1.025 | +0.73 |
| train_2/denver_8 | 0.004370 | −0.94 | **−1.92** | −1.00 | −0.460 → −0.968 | +0.60 |
| train_8/denver_6 | 0.003231 | +0.05 | +0.04 | −0.01 | −0.044 → −0.058 | −0.02 |
| train_1/fairbanks_1 | 0.045730 | −0.21 | **−0.51** | −0.30 | −2.189 → −3.291 | +1.87 |
| train_2/fairbanks_2 | 0.012412 | −0.46 | **−1.12** | −0.66 | −0.514 → −1.053 | +1.09 |
| train_4/fairbanks_4 | 0.001606 | −0.28 | +1.09 | +1.37 | −0.289 → −0.458 | +0.06 |
| train_5/fairbanks_5 | 0.001553 | +0.00 | −0.54 | −0.54 | −0.086 → −0.121 | +0.02 |
| train_7/fairbanks_7 | 0.008450 | −0.36 | −0.99 | −0.63 | −0.469 → −1.089 | +0.64 |
| train_8/fairbanks_8 | 0.024166 | −0.26 | **−1.04** | −0.79 | −0.336 → −0.910 | +2.11 |
| train_1/phoenix_5 | 0.014240 | +0.00 | +0.00 | −0.00 | 0.000 → 0.000 | +0.00 |
| train_2/phoenix_6 | 0.007199 | −0.00 | +0.00 | +0.00 | −0.000 → −0.000 | +0.00 |
| train_4/phoenix_8 | 0.001644 | +0.37 | +1.34 | +0.97 | −0.221 → −0.343 | +0.04 |
| train_5/phoenix_1 | 0.002350 | −0.26 | +0.73 | +0.99 | −0.891 → −1.344 | +0.18 |
| train_7/phoenix_3 | 0.005454 | +0.00 | +0.00 | −0.00 | −0.000 → −0.000 | +0.00 |
| train_8/phoenix_4 | 0.009606 | −0.00 | +0.00 | +0.00 | 0.000 → 0.000 | +0.00 |
| train_1/singapore_8 | 0.006410 | +0.00 | +0.00 | +0.00 | −0.002 → −0.002 | +0.00 |
| train_8/singapore_7 | 0.005342 | +0.00 | +0.00 | +0.00 | −0.000 → −0.000 | +0.00 |

| subset | CtrlW | CtrlW2 v4 | CtrlW4 | CtrlW4 vs v4 | better | worst | comfort | energy | carbon | h06 sum |
|---|---|---|---|---|---|---|---|---|---|---|
| **12** (fbk+phx) | 0.011201 | 0.011180 (−0.18%) | **0.011146 (−0.49%)** | −0.30% | 5/12 | +1.34% | −6.21 → −10.00 | −502.86 → −499.75 (+0.6%) | −652.36 → −646.02 (+1.0%) | −4.996 → −8.610 |
| **10** (VoI subset) | 0.011920 | 0.011896 (−0.20%) | **0.011856 (−0.54%)** | −0.34% | 4/10 | +0.69% | −5.33 → −8.34 | −449.97 → −447.20 (+0.6%) | −574.70 → −568.91 (+1.0%) | −3.886 → −6.886 |

The lift's own contribution reproduces the VoI study on its own 10-scenario
subset: `results/voi_night_fixed` (CtrlW + the same fixed lift, its claim of
−0.50%) scores 0.011860 there, CtrlW4 on the CtrlW2 base scores 0.011856 — the
two levers are additive to within 0.03% of the subset.

**dev30 estimate.**  18 of the 30 scenarios measured; if the other 12 stay at
their CtrlW2 v4 values the dev30 score is **0.0073415 (−0.22% vs v4, −0.38% vs
CtrlW)**.  The 12 unmeasured ones are the remaining atlanta/denver/singapore
buildings, where 4 of the 6 measured atlanta/denver scenarios respond (mixed
signs) and singapore is inert, so the estimate is on the cautious side.

### Caveats and why this is still a GO

* **The comfort bill is large and must be reported, not hidden.**  Comfort regret
  over the 18 scenarios: −8.90 → −13.50 (v4 → CtrlW4), hour-06 comfort −7.05 →
  −11.29; against CtrlW the 18-scenario sum goes −8.27 → −13.50.  The lift spends
  more heat at 02:00-06:00 than it saves there
  (03-06 energy +5.3 to +6.0 on the subsets) and wins it back later in the day
  at a lower marginal price, so total energy −0.6% and carbon −1.0% while the
  building is *warmer* overnight and *colder* at arrival in the lift-fired zones.
  The reward weights comfort 0.6 but the productive-regret term is normalised by
  its own scale, so the energy/carbon terms pay for it — that is the mechanism,
  and it is the same trade the VoI study measured on the CtrlW base (−0.50%
  score, comfort −31% there).
* **Two warm scenarios misfire** (train_4/phoenix_8 +1.34%, train_5/phoenix_1
  +0.73%): warm zones that happen to sit on the 282 K floor at night get lifted
  and then re-cooled.  Both are far below the 3% guard, and the same gate makes
  4 of the 6 warm scenarios exactly inert.  If dev30 shows this pattern on more
  warm buildings, `night_lift_lead_hours` (4 h) and the at-floor tolerance are
  the knobs to re-tune; neither was swept on top of v4.
* The `night_lift_K = 2.0` value is inherited from the VoI study's forecast
  ablation, not re-tuned for the CtrlW2 stack; a sweep (1/2/3 K) is a cheap
  follow-up if the dev30 result holds.
* Two hosts again: `ctw4_sub18` is Mac + `--fast`; the CtrlW2 v4 baselines are the
  PC/Mac mix described above.  Machine noise is 4e-6 relative, the effect is 5e-3.

## Files

* `policies/ctrl_w2.py` — CtrlW2, defaults = v4, docstring rewritten with the
  three-variant table, the corrected phoenix finding and the reproducibility note.
* `policies/ctrl_w4.py` — CtrlW4 = CtrlW2 + the fixed pre-dawn night lift
  (submission candidate, defaults are the measured configuration).
* `results/ctw2_v2` (12), `results/ctw2_trim` (12, PC), `results/ctw2_v4` (12, PC),
  `results/ctw2_v4_dev30` (18, Mac), `results/ctw2_trim_check` (2, Mac
  reproducibility), `results/w2off_a`/`w2off_b` (400-step adaptive-off parity),
  `results/ctw4_sub18` (18, Mac, the CtrlW4 evaluation),
  `results/w4chk_*` (2×576 steps: CtrlW2 / lift-off / lift-on / StackLift).
* `scratch/AdaptivePreheat/compare2.py` — subset comparison across any number of
  variants (score, Δ, hour-06 comfort, 03-06 energy, component sums);
  `--only` pins an explicit scenario list, `label=dir+dir` unions result dirs.
* `scratch/AdaptivePreheat/dev30.py` — dev30 score + per-scenario/by-term diff,
  accepts `label=dir+dir` so a set split across machines is scored exactly.
* `scratch/AdaptivePreheat/stack_lift.py` — exactness harness: OracleFcstCtrlW's
  unmodified `_night_lift` stacked on CtrlW2 (analysis-only, imports an oracle).
* `scratch/AdaptivePreheat/diag.py`, `analysis.py`, `diag/`, `smoke/`,
  `trimprobe/` — the per-step diagnosis behind the docstring's mechanism section.
* `scratch/AdaptivePreheat/pc_v4.sh`, `pc_v4_dev30.sh`, `run_chunks.sh` — the runs.
