## Decomposition of the energy+carbon loss (reward units, positive)

| scenario | station | $\bar T_\mathrm{out}$ (°C) | comfort | energy | carbon | total | heating: outside-air | heating: envelope/internal | boiler standby | cooling: outside-air | cooling: internal/solar | fans | pump | residual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fairbanks_1 | Fairbanks | -21.4 | 2.607 | 112.25 | 280.30 | 392.55 | 69.65 | 306.69 | 5.08 | -0.00 | 0.00 | 8.618 | 2.508 | 0.000 |
| fairbanks_2 | Fairbanks (large) | -8.2 | 0.569 | 30.27 | 76.42 | 106.69 | 9.29 | 90.55 | 2.12 | -0.00 | 0.00 | 4.062 | 0.661 | 0.000 |
| denver_5 | Denver | 24.2 | 0.575 | 29.33 | 8.06 | 37.39 | 0.00 | 0.16 | 1.56 | 4.89 | 27.81 | 2.968 | 0.000 | 0.000 |
| atlanta_3 | Atlanta | 20.8 | 0.224 | 19.18 | 5.39 | 24.58 | 0.01 | 0.39 | 0.91 | 1.24 | 18.20 | 3.830 | 0.000 | 0.000 |
| phoenix_4 | Phoenix | 34.4 | -0.000 | 67.46 | 15.54 | 83.00 | -0.00 | -0.00 | 0.07 | 17.43 | 60.63 | 4.869 | -0.000 | 0.000 |
| singapore_3 | Singapore | 28.6 | 0.416 | 47.66 | 10.99 | 58.65 | -0.00 | -0.00 | 0.08 | 6.72 | 48.33 | 3.512 | -0.000 | 0.000 |

| scenario | station | coil heat (kWh$_{th}$) | ventilation-carried (kWh$_{th}$, %) | marginal coil relief, 100% recirc (%) | chiller (kWh$_e$) | vent share of chiller (%) | supply fan (kWh$_e$) | exhaust fan (kWh$_e$) | pump (kWh$_e$) |
|---|---|---|---|---|---|---|---|---|---|
| fairbanks_1 | Fairbanks | 377.2 | 190.4 (50%) | 18.6% | 0.00 | 0.00 (0%) | 3.42 | 1.03 | 1.30 |
| fairbanks_2 | Fairbanks (large) | 227.7 | 142.6 (63%) | 9.3% | 0.00 | 0.00 (0%) | 3.55 | 1.06 | 0.76 |
| denver_5 | Denver | 0.2 | 2.4 (--) | 3.2% | 15.42 | 2.30 (15%) | 1.01 | 0.40 | 0.00 |
| atlanta_3 | Atlanta | 0.9 | 22.7 (--) | 3.8% | 20.71 | 1.31 (6%) | 2.56 | 1.57 | 0.00 |
| phoenix_4 | Phoenix | 0.0 | 0.3 (--) | -- | 28.07 | 6.22 (22%) | 1.33 | 0.43 | 0.00 |
| singapore_3 | Singapore | 0.0 | 0.0 (--) | -- | 39.30 | 4.78 (12%) | 1.92 | 0.59 | 0.00 |

## Rollout metadata and the reconciliation check

| station | scenario | steps | zones | AHUs | mean reward/step | recirculation | coil saturation | energy rel. err | carbon rel. err |
|---|---|---|---|---|---|---|---|---|---|
| Fairbanks | building_train_1/fairbanks_1 | 8641 | 41 | 4 | -0.045730 | 0.700 | 0.38 | 7.6e-10 | 3.8e-10 |
| Fairbanks (large) | building_train_2/fairbanks_2 | 8641 | 87 | 9 | -0.012412 | 0.700 | 0.14 | 6.6e-10 | 3.0e-10 |
| Denver | building_train_7/denver_5 | 8641 | 38 | 4 | -0.004394 | 0.671 | 0.01 | 3.9e-10 | 5.4e-10 |
| Atlanta | building_train_2/atlanta_3 | 8641 | 87 | 9 | -0.002870 | 0.608 | 0.01 | 5.0e-10 | 4.4e-10 |
| Phoenix | building_train_8/phoenix_4 | 8641 | 32 | 3 | -0.009606 | 0.692 | nan | 3.6e-10 | 7.2e-10 |
| Singapore | building_train_4/singapore_3 | 8641 | 56 | 6 | -0.006835 | 0.699 | nan | 2.6e-10 | 1.5e-10 |

## Key numbers

- Across the six scenarios the energy+carbon loss is 702.8 reward units: heating 486.6 (69%), of which the marginal ventilation duty is 79.0 (11% of the total, 16% of the heating); cooling 185.3 (26%), of which ventilation 30.3 (16% of the cooling); fans 27.9 (4%); pump 3.17 (0.5%).
- Fairbanks: the outside-air stream at 30% of supply airflow carries 190.4 kWh_th into the building over the episode, 50% of the 377.2 kWh_th the reheat coils delivered; recirculating everything would cut the coil duty by 18.6% only, because the coils are water-limited (median duty 38% of the air side's capacity).
- Fairbanks (large): the outside-air stream at 30% of supply airflow carries 142.6 kWh_th into the building over the episode, 63% of the 227.7 kWh_th the reheat coils delivered; recirculating everything would cut the coil duty by 9.3% only, because the coils are water-limited (median duty 14% of the air side's capacity).
- Denver: the outside-air stream at 33% of supply airflow carries 2.4 kWh_th into the building over the episode, against 0.2 kWh_th of reheat, because almost no heating ran; recirculating everything would cut the coil duty by 3.2% only, because the coils are water-limited (median duty 1% of the air side's capacity).
- Atlanta: the outside-air stream at 39% of supply airflow carries 22.7 kWh_th into the building over the episode, against 0.9 kWh_th of reheat, because almost no heating ran; recirculating everything would cut the coil duty by 3.8% only, because the coils are water-limited (median duty 1% of the air side's capacity).
- Phoenix: the outside-air stream at 31% of supply airflow carries 0.3 kWh_th into the building over the episode, against 0.0 kWh_th of reheat, because almost no heating ran.
- Singapore: the outside-air stream at 30% of supply airflow carries 0.0 kWh_th into the building over the episode, against 0.0 kWh_th of reheat, because almost no heating ran.
- Fairbanks: the chiller draws 0.00 kWh_e, of which 0% is attributable to the outside air (the rest is internal, solar and envelope gain); the exhaust fan, which exists only to make room for that outside air, adds 1.03 kWh_e against 3.42 kWh_e for the supply fan.
- Fairbanks (large): the chiller draws 0.00 kWh_e, of which 0% is attributable to the outside air (the rest is internal, solar and envelope gain); the exhaust fan, which exists only to make room for that outside air, adds 1.06 kWh_e against 3.55 kWh_e for the supply fan.
- Denver: the chiller draws 15.42 kWh_e, of which 15% is attributable to the outside air (the rest is internal, solar and envelope gain); the exhaust fan, which exists only to make room for that outside air, adds 0.40 kWh_e against 1.01 kWh_e for the supply fan.
- Atlanta: the chiller draws 20.71 kWh_e, of which 6% is attributable to the outside air (the rest is internal, solar and envelope gain); the exhaust fan, which exists only to make room for that outside air, adds 1.57 kWh_e against 2.56 kWh_e for the supply fan.
- Phoenix: the chiller draws 28.07 kWh_e, of which 22% is attributable to the outside air (the rest is internal, solar and envelope gain); the exhaust fan, which exists only to make room for that outside air, adds 0.43 kWh_e against 1.33 kWh_e for the supply fan.
- Singapore: the chiller draws 39.30 kWh_e, of which 12% is attributable to the outside air (the rest is internal, solar and envelope gain); the exhaust fan, which exists only to make room for that outside air, adds 0.59 kWh_e against 1.92 kWh_e for the supply fan.
