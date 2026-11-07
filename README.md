# Stop Heating an Empty Building

Code, per-scenario results and paper source for *Stop Heating an Empty Building: Reward Anatomy
and a Supervisory Layer for the NeurIPS 2026 Smart Buildings Challenge* (Semyon Morev, NUST MISIS),
a Round-1 paper of the [NeurIPS 2026 Smart Buildings Challenge](https://www.kaggle.com/competitions/neurips-2026-smart-buildings).

The paper (`paper/main.pdf`) adds a supervisory layer to the strongest public controller and finds
that what is left to fix is the schedule of the plant, not the control law. CtrlW5 stops heating
the empty building at night and delivers the morning recovery in one burst at the coil's cap.

| controller | dev30, shipped | dev30, corrected | Kaggle public |
|---|---|---|---|
| CtrlW | 0.0073694 | 0.0071449 | 0.01051 |
| CtrlW5-278 (278 K frost floor, the paper's controller) | 0.0071980 | 0.0070409 | 0.01032 |
| CtrlW5-262 (no frost floor) | 0.0071338 | 0.0069430 | 0.01024 |

Scores are the challenge score, lower is better. dev30 is the 30 shipped training scenarios with
30-day episodes. The corrected tree is the shipped simulator plus starter-kit PR #2 and PR #3.

## Layout

| path | contents |
|---|---|
| `policies/` | controllers |
| `sbx/` | experiment harness |
| `cenv/` | the simulator step in C, bit-identical to the Python step (`--cenv fused`); gates in `cenv/verify/` |
| `diffsim/` | an exact differentiable twin of the step |
| `fastsim/` | a reduced-order thermal model, used as a ranking filter |
| `kaggle/` | the submission notebook builder and a remote runner for evaluations on Kaggle CPU kernels |
| `results/` | one JSON per scenario plus a `summary.json` per run |
| `docs/` | the study reports behind the paper |
| `data/weather_ext/` | external weather strips derived from Meteostat |
| `paper/` | LaTeX source, figure scripts and the PDF |
| `gpu/` | ROCm/WSL2 scripts for the GPU solver study |

More detail on the main directories:

- **`policies/`.**
  - `wu_core.py` is the public base controller (Daniel Wu, Apache-2.0).
  - `ctrl_w.py` to `ctrl_w7.py` are ours: `ctrl_w5.py` is the paper's controller; `ctrl_w6.py` and `ctrl_w7.py` keep refuted variants reproducible.
  - `exploit_b1.py`, `exploit_b2.py` and `oracle_*.py` are analysis-only probes and were never submitted.
- **`sbx/`.**
  - `evalrun.py`: parallel scenario runner;
  - `scenarios.py`: scenario sets;
  - `make_sims.sh`: builds the shipped and corrected simulator trees;
  - `make_pool.py`: generated buildings;
  - `tune.py`: CMA-ES;
  - `integrity.py`, `energyflow.py`: accounting instruments;
  - `fastpatch.py`: exact pure-Python speed-ups.
- **`results/`.** Every number in the paper is read from these files; the `% src:` comments in `paper/` name the directories.

## Setup

Python 3.12 (3.13 breaks numba).

```bash
git clone https://github.com/MorS1337/sb26-stop-heating.git && cd sb26-stop-heating
git clone https://github.com/smart-buildings-2026/starter-kit.git
git -C starter-kit checkout 95cea5b
python3.12 -m venv starter-kit/.venv
starter-kit/.venv/bin/pip install -e starter-kit cma
bash sbx/make_sims.sh                          # sims/shipped (kit 95cea5b) and sims/corrected (+ PR #2, #3)
starter-kit/.venv/bin/python -m sbx.make_pool  # generated buildings and weather strips
for t in shipped corrected; do
  starter-kit/.venv/bin/python sbx/weather_ext.py --copy-to sims/$t/smart_buildings/weather
done
```

## Reproduce

```bash
PY=starter-kit/.venv/bin/python
# Table 1: CtrlW5-278 on the 30 shipped scenarios (--sim corrected for the corrected tree)
$PY -m sbx.evalrun --policy policies.ctrl_w5:CtrlW5 --params '{"standby_minimum": 278.0}' \
    --set dev30 --steps 8641 --sim shipped --workers 4 --cenv fused --fast --tag ctw5frost_dev30_shipped
# the 49 holdouts
$PY -m sbx.evalrun --policy policies.ctrl_w5:CtrlW5 --params '{"standby_minimum": 278.0}' \
    --set gen_hold ext_hold --steps 8641 --workers 4 --cenv fused --fast --tag ctw5frost_holdouts_shipped
# the seven-day anchor (CtrlW5 without the floor: -80.306)
$PY -m sbx.evalrun --policy policies.ctrl_w5:CtrlW5 --scenario building_train_1/fairbanks_1 \
    --steps 2016 --workers 1 --cenv fused --fast --tag anchor_ctw5
# exactness of the C step against the Python step
$PY -m cenv.verify.gate trace --scenario building_train_8/fairbanks_8 --steps 600 --out /tmp/t8_py
$PY -m cenv.verify.gate trace --scenario building_train_8/fairbanks_8 --steps 600 --cenv fused --out /tmp/t8_c
$PY -m cenv.verify.gate compare /tmp/t8_py.npz /tmp/t8_c.npz
# the leaderboard notebook
$PY kaggle/build_notebook.py --policy policies/ctrl_w5.py:CtrlW5 --param standby_minimum=278.0 --slug sbx-ctrl-w5f
```

On one machine, `--cenv fused --fast` gives the same totals as plain Python at the scenarios'
default start. With `--start-offset-days` other than 0, leave `--cenv` off: this is a known issue
in the fused step, and a fix is in progress.

Across machines (arm64 with Accelerate against x86-64 with OpenBLAS), 30-day totals differ by up
to a few 1e-5 relative: BLAS round-off is amplified by the closed loop. Compare runs from one
machine.

## License

- The code is Apache-2.0; see `LICENSE` and, for attributions, `NOTICE`.
- `data/weather_ext/` is CC BY 4.0, derived from Meteostat.
- The competition data and the simulator are not redistributed; `sbx/make_sims.sh` rebuilds them from the public starter kit.
