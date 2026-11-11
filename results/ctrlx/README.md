# CtrlX result directories

One JSON per scenario (`sbx.evalrun` output) plus `summary.json`, for every CtrlX number in
`paper/main.tex`. The experiment log that names each directory is `docs/ctrlx_experiments.md`.

Paired 30-day runs (8641 steps, default occupancy seed) ran on one machine (Ryzen 5 5600X, WSL2,
fused C step); `bench7_*`, `integrity_*` and `ctrw_a_rerun_*` ran on an Apple M3. Compare runs only
within one machine.

| directory | what |
|---|---|
| `arx_p7f_w5_dev30_{corrected,shipped}` | CtrlW5-278, the paired baseline |
| `arx_wrc_x_dev30_corrected` | CtrlX with mechanisms 1-11 (late-start gate 4 K), the ablation reference |
| `arx_abl_m{1..11}_dev30_corrected` | leave-one-out: mechanism *n* switched off |
| `arx_abl_gap6_dev30_corrected` | mechanisms 1-11, gate 6 K ("v5") |
| `arx_op_op310_dev30_corrected` | + mechanism 12, off-peak hot-water floor ("v6") |
| `arx_sz_both05_dev30_corrected` | + mechanism 13, size-aware margin and lead: **CtrlX** ("v7") |
| `arx_sz_{sl05,sm05}_dev30_corrected` | mechanism 13, lead half and margin half alone |
| `arx_ld_{l075,l100}_dev30_corrected` | uniform arrival lead 0.75 h / 1.0 h on v6 |
| `arx_seed{12345,777}_{x,b}_dev30_corrected` | reseeded occupancy: v5 (`x`) against the controller without mechanisms 10-11 (`b`) |
| `arx_hc_{w5,v3,v5,x,v7}_gen_hold-ext_hold_corrected` | 49 holdouts, corrected tree: CtrlW5-278, mechanisms 1-9, v5, v6, CtrlX |
| `arx_p7a_x_dev30_shipped`, `arx_p7b_x_gen_hold-ext_hold_shipped` | CtrlX on the shipped tree |
| `arx_p7c_tank_dev30_shipped`, `arx_p7d_tank_dev30_corrected`, `arx_p7e_tank_gen_hold-ext_hold_shipped` | CtrlX + tank hold (`water_idle_hold`, `water_peak_hold_hours=30`, `water_rise_guard_K=2.5`, `water_offpeak_floor=0`) |
| `arx_v5s_gap6_dev30_shipped`, `arx_ops_x_dev30_shipped`, `arx_v6h_x_gen_hold-ext_hold_shipped` | v5 and v6 on the shipped tree |
| `arx_v6s_hpk_dev30_shipped`, `arx_ops_hpk_rg25_dev30_shipped`, `arx_v6h_hpk_gen_hold-ext_hold_shipped` | earlier tank-hold variants (v5 and v6 code) |
| `bench7_corrected_{gap6,op310,both05}` | the development benchmark (10 scenarios, 2016 steps, seed 12345) at v5, v6, CtrlX |
| `bench7_{corrected,shipped}_rg25`, `bench7_shipped_v6_off` | the setpoint cap (`water_rise_guard_K=2.5`) on both trees, and its shipped reference |
| `bench7_seed_{v6,v7}_{777,2024}` | the benchmark scenarios with two other occupancy seeds |
| `integrity_anchor_weeks_{v5,v5hpk,v7,v7tank}` | gas billed against heat drawn (`sbx.integrity`), 7 days, shipped tree |
| `ctrw_a_rerun_dev30_shipped` | ablation row (a): the base controller with CtrlW's occupancy release only |

## Policy modules named in `summary.json`

The paired 30-day runs imported snapshot copies of `policies/ctrl_x.py`, so that a queued run kept
its code while development continued. The `policy` field of each `summary.json` names the snapshot,
and the `params` field lists every override passed to it.

| module | code |
|---|---|
| `policies.ctrl_x_wr`, `policies.ctrl_x_v5` | v4: mechanisms 1–11, late-start gate 4 K |
| `policies.ctrl_x_v6` | v5 plus the tank options, off by default |
| `policies.ctrl_x_v8` | v6: v5 plus the off-peak hot-water floor (mechanism 12) |
| `policies.ctrl_x_v9` | the v7 code with the size powers off by default (`size_lead_power`, `size_margin_power` = 0) |
| `policies.ctrl_x_v10` | v7, the published `policies/ctrl_x.py` (Kaggle submission 58217472) |
| `policies.ctrl_x` | the repository module at the time of the run; `bench7_*` and `integrity_*` list their overrides in `params` |

Every mechanism after v4 was added behind a `PARAMS` switch, so each configuration above can be rebuilt
from the published file by switching the later mechanisms off: `size_lead_power` and `size_margin_power`
0 for v6, plus `water_offpeak_floor` 0 for v5, plus `late_start_gap_K` 4 for v4. These rebuilds were not
re-run bit for bit. The v7 corrected-tree rows need no rebuild: `arx_sz_both05` and `arx_hc_v7` pass the
two size powers 0.5 to `ctrl_x_v9`, which matches the published defaults.

Better / identical / worse counts in the paper treat two scores as identical when they differ by at most
0.005 %. The Kaggle submissions behind the leaderboard numbers are listed in `../kaggle_submissions.csv`.
