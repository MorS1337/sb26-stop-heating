# External weather set (unseen-climate transfer validation)

The graded Kaggle run uses weather stations that appear nowhere in the shipped
training set (shipped: `fairbanks`, `denver`, `atlanta`, `phoenix`, `singapore`;
`minneapolis` ships unused). This directory holds a 2023 hourly climate set for
17 stations the policy has never seen, converted into the simulator's exact
weather-CSV schema plus drop-in preset YAMLs, so we can measure transfer to
unseen climates instead of only to unseen buildings.

Everything here is produced by `sbx/weather_ext.py`; nothing is written into
`starter-kit/`.

```
data/weather_ext/
  weather/<station>_<n>.yaml   drop-in presets (3 per station, 51 total)
  weather/csv/<station>.csv    simulator-format 2023 hourly weather
  raw/<id>.csv.gz              cached Meteostat bulk downloads
  report.json                  gaps, annual stats, coefficients, strips
```

## Source and license

* Source: Meteostat v2 bulk hourly endpoint,
  `https://bulk.meteostat.net/v2/hourly/<station_id>.csv.gz`, free and
  key-less. Meteostat aggregates SYNOP/METAR reports (NOAA ISD and national
  weather services) and publishes the result as hourly time series in UTC:
  `date,hour,temp,dwpt,rhum,prcp,snow,wdir,wspd,wpgt,pres,tsun,coco`, with
  `temp`/`dwpt` in degC, `rhum` in percent, `wspd` in km/h.
* License: Meteostat API data is **CC BY 4.0** (code MIT). Attribution:
  "Meteostat (2026), https://meteostat.net". Underlying observations are public
  weather-service / NOAA ISD reports.
* Station metadata (timezone, inventory) from
  `https://bulk.meteostat.net/v2/stations/full.json.gz`.

## Stations

17 stations in 2023, hourly, local station clock time. `h` is the preset's
`convection_coefficient` (see "Convection coefficient" below).

| slug | Meteostat id | station | tz | climate | mean °C | min–max °C | v̄ m/s | h W/m²K |
|---|---|---|---|---|---|---|---|---|
| `chicago` | 72530 | Chicago O'Hare Airport | America/Chicago | continental-humid | 12.2 | -17.8–37.2 | 4.0 | 23.3 |
| `anchorage` | 70273 | Anchorage Ted Stevens Airport | America/Anchorage | subarctic | 3.0 | -20.6–22.2 | 2.8 | 16.5 |
| `montreal` | 71627 | Montreal International Airport | America/Toronto | continental-cold | 8.9 | -29.3–34.0 | 4.3 | 24.5 |
| `helsinki` | 02974 | Helsinki-Vantaa | Europe/Helsinki | nordic | 6.3 | -17.4–29.4 | 4.1 | 23.7 |
| `moscow` | UUEE0 | Moscow Sheremetyevo | Europe/Moscow | severe-continental | 6.8 | -25.0–31.0 | 4.5 | 25.4 |
| `beijing` | 54511 | Beijing | Asia/Shanghai | continental-monsoon | 13.9 | -21.0–42.0 | 2.6 | 15.3 |
| `reykjavik` | 04018 | Keflavikurflugvollur | Atlantic/Reykjavik | subpolar-marine | 5.2 | -10.0–18.0 | 6.9 | 27.0 |
| `lasvegas` | 72386 | McCarran International Airport | America/Los_Angeles | hot-arid | 20.8 | -0.6–45.6 | 3.1 | 18.2 |
| `dubai` | 41194 | Dubai International Airport | Asia/Dubai | hot-arid-coastal | 30.1 | 14.0–46.0 | 3.7 | 21.2 |
| `houston` | 72243 | Houston Intercontinental | America/Chicago | hot-humid | 22.2 | 0.6–42.8 | 3.4 | 19.6 |
| `miami` | 72202 | Miami International Airport | America/New_York | subtropical-humid | 26.3 | 7.2–36.1 | 3.1 | 18.1 |
| `bangkok` | 48456 | Don Muang | Asia/Bangkok | tropical-monsoon | 29.5 | 18.3–40.0 | 3.1 | 18.5 |
| `tokyo` | 47671 | Tokyo International Airport | Asia/Tokyo | humid-subtropical | 18.3 | -2.0–37.0 | 5.2 | 27.0 |
| `seattle` | 72793 | Seattle-Tacoma Airport | America/Los_Angeles | marine-mild | 11.9 | -5.6–35.0 | 3.3 | 19.5 |
| `sanfrancisco` | 72494 | San Francisco Airport | America/Los_Angeles | cool-marine | 14.8 | 3.3–32.8 | 4.5 | 25.8 |
| `london` | 03772 | London Heathrow Airport | Europe/London | oceanic-temperate | 12.4 | -8.3–32.5 | 4.2 | 23.9 |
| `auckland` | 93119 | Auckland Airport | Pacific/Auckland | southern-marine | 15.9 | 4.0–27.0 | 4.9 | 27.0 |

Seasonal inversion is covered (Auckland), as are the extremes the shipped five
miss: a 46 °C desert week (Dubai, Las Vegas), a 40 °C monsoon week (Bangkok),
-29 °C (Montreal) and a 6.9 m/s windy marine site (Reykjavik).

## Gap statistics

Gap fraction is the share of the 2023 local-year hours with no observation,
measured **before** interpolation:

| variable | worst station | value |
|---|---|---|
| `temp` | all 17 | 0.0000 |
| `dwpt` | all 17 | 0.0000 |
| `rhum` | all 17 | 0.0000 |
| `wspd` | all 17 | 0.0000 |

Meteostat's v2 hourly series for major airports is complete for 2023 (it merges
several upstream feeds), so no station was dropped by the >5 % rule
(`MAX_GAP`). The converter still interpolates linearly in time and edge-fills;
it also clamps `rhum` to [0, 100], forces `wspd >= 0` and forces
`dwpt <= temp`, then asserts no NaN survives — a NaN would poison every
downstream observation.

Row counts are 8760 for stations without DST and 8759 for those with it: the
duplicated fall-back hour is averaged so `Time` labels stay unique, and the
spring-forward hour genuinely does not exist on a wall clock.

## Simulator CSV contract (what the converter has to satisfy)

Read from the shipped code, not guessed:

* `smart_control/simulator/weather_controller.py::read_weather_csv` drops the
  unnamed index column, parses `Time` as `%Y%m%d-%H%M` **with `utc=True`** and
  indexes by seconds since epoch; `np.interp` then interpolates `TempF` and
  `Humidity` (that is all the environment consumes — `TempC`, `DewPointC`,
  `WindspeedKmph`, `StationName`, `Location` are carried for humans).
* The shipped CSVs are in **local station clock time** despite the loader
  calling it UTC: their diurnal minimum sits at 03:00–06:00 and the maximum at
  13:00–15:00, which is only possible for local stamps. We reproduce that
  convention, because the policy sees hour-of-day features and the building's
  solar gains are aligned in that frame.
* A preset YAML is `name`, `weather_csv` (path **relative to the package
  `weather/` dir**, so `csv/<slug>.csv`), `convection_coefficient`,
  `start_timestamp` (`%Y-%m-%d %H:%M`, episode start, local frame).
* `ReplayWeatherController` raises if a lookup falls outside the CSV's
  timestamps, so every strip's 30-day window must end before the last label
  (2023-12-31 23:00).

## Strips

30-day episodes starting Sunday 06:00 local (8640 five-minute steps), months
chosen to stress each climate: deep winter, peak summer, shoulder.
`start_timestamp` is the Sunday on or after the 8th of the month.

| preset | start (local) | window T min–max °C | mean RH % |
|---|---|---|---|
| `chicago_1` | 2023-01-08 06:00 | -17.8–12.8 | 71 |
| `chicago_2` | 2023-07-09 06:00 | 17.2–33.3 | 64 |
| `chicago_3` | 2023-10-08 06:00 | -1.7–27.8 | 66 |
| `anchorage_1` | 2023-01-08 06:00 | -17.2–5.6 | 85 |
| `anchorage_2` | 2023-07-09 06:00 | 9.4–22.2 | 81 |
| `anchorage_3` | 2023-10-08 06:00 | -6.7–8.3 | 77 |
| `montreal_1` | 2023-01-08 06:00 | -29.3–2.0 | 76 |
| `montreal_2` | 2023-07-09 06:00 | 11.7–29.0 | 74 |
| `montreal_3` | 2023-10-08 06:00 | -2.7–20.0 | 74 |
| `helsinki_1` | 2023-01-08 06:00 | -14.5–3.7 | 91 |
| `helsinki_2` | 2023-07-09 06:00 | 5.9–29.4 | 76 |
| `helsinki_3` | 2023-10-08 06:00 | -3.6–13.8 | 84 |
| `moscow_1` | 2023-01-08 06:00 | -20.0–4.0 | 82 |
| `moscow_2` | 2023-07-09 06:00 | 11.0–31.0 | 75 |
| `moscow_3` | 2023-10-08 06:00 | -3.0–15.0 | 86 |
| `beijing_1` | 2023-01-08 06:00 | -20.0–13.0 | 38 |
| `beijing_2` | 2023-07-09 06:00 | 20.0–39.0 | 74 |
| `beijing_3` | 2023-10-08 06:00 | -2.2–28.0 | 62 |
| `reykjavik_1` | 2023-01-08 06:00 | -9.8–8.2 | 80 |
| `reykjavik_2` | 2023-07-09 06:00 | 7.0–18.0 | 79 |
| `reykjavik_3` | 2023-10-08 06:00 | -1.0–11.0 | 80 |
| `lasvegas_1` | 2023-04-09 06:00 | 11.1–36.1 | 18 |
| `lasvegas_2` | 2023-07-09 06:00 | 23.9–45.6 | 16 |
| `lasvegas_3` | 2023-09-10 06:00 | 12.8–36.1 | 28 |
| `dubai_1` | 2023-01-08 06:00 | 14.0–29.0 | 61 |
| `dubai_2` | 2023-07-09 06:00 | 31.4–46.0 | 48 |
| `dubai_3` | 2023-09-10 06:00 | 29.2–45.0 | 51 |
| `houston_1` | 2023-02-12 06:00 | 1.1–30.0 | 75 |
| `houston_2` | 2023-07-09 06:00 | 23.9–39.4 | 65 |
| `houston_3` | 2023-09-10 06:00 | 12.2–35.6 | 71 |
| `miami_1` | 2023-01-08 06:00 | 7.2–30.6 | 71 |
| `miami_2` | 2023-04-09 06:00 | 20.0–32.2 | 71 |
| `miami_3` | 2023-07-09 06:00 | 23.9–36.1 | 71 |
| `bangkok_1` | 2023-04-09 06:00 | 24.0–40.0 | 65 |
| `bangkok_2` | 2023-09-10 06:00 | 24.0–35.0 | 79 |
| `bangkok_3` | 2023-11-12 06:00 | 21.0–35.0 | 66 |
| `tokyo_1` | 2023-01-08 06:00 | -2.0–16.0 | 57 |
| `tokyo_2` | 2023-08-13 06:00 | 21.0–34.0 | 81 |
| `tokyo_3` | 2023-10-08 06:00 | 13.0–27.0 | 68 |
| `seattle_1` | 2023-01-08 06:00 | -4.4–15.0 | 75 |
| `seattle_2` | 2023-07-09 06:00 | 12.2–32.2 | 60 |
| `seattle_3` | 2023-10-08 06:00 | 1.1–22.2 | 78 |
| `sanfrancisco_1` | 2023-01-08 06:00 | 3.3–20.0 | 69 |
| `sanfrancisco_2` | 2023-07-09 06:00 | 12.2–26.1 | 74 |
| `sanfrancisco_3` | 2023-09-10 06:00 | 12.8–32.8 | 70 |
| `london_1` | 2023-01-08 06:00 | -8.3–12.7 | 82 |
| `london_2` | 2023-07-09 06:00 | 10.3–24.4 | 73 |
| `london_3` | 2023-10-08 06:00 | 2.3–25.1 | 84 |
| `auckland_1` | 2023-01-08 06:00 | 15.0–26.0 | 82 |
| `auckland_2` | 2023-07-09 06:00 | 4.0–17.0 | 84 |
| `auckland_3` | 2023-10-08 06:00 | 9.0–22.0 | 80 |

All 51 windows are inside their CSV (verified by `strip_window_stats` in
`report.json`: `covers_window: true`, hourly rows, latest window end
2023-12-12). Seven strips cross a DST transition on purpose, so the wall-clock
handling above is exercised rather than assumed: `houston_1` crosses the US
spring forward (2023-03-12), and `chicago_3`, `anchorage_3`, `montreal_3`,
`seattle_3` (US fall back 2023-11-05), `london_3` and `helsinki_3` (EU fall
back 2023-10-29) cross a fall back, where the duplicated hour is averaged.

## Convection coefficient

The eight strips shipped per station share one constant coefficient per station
(fairbanks 8.8, singapore 17.6, atlanta 19.1, phoenix 19.5, minneapolis 21.5,
denver 26.7 W/m²/K). No generator deriving it is shipped, and it is not a
function of the wind in the file for any simple law we tested. Over the six
shipped stations it tracks annual mean wind speed `v` [m/s] as

    h = 1.72 + 5.32 * v      (least squares on the shipped wind columns;
                              RMSE 2.0, max error 2.3 W/m²/K)

so we use exactly that fit, clipped to the shipped range [8.0, 27.0] and
rounded to 0.1. Our stations therefore sit on the same manifold as the shipped
ones rather than inventing a new convention: calm interiors (Beijing 15.3) are
below windy coasts (Reykjavik/Tokyo/Auckland 27.0).

## Verification (this machine, CPython 3.12.8, starter-kit venv)

1. **Simulator load.** One simulator process at a time. Scratch package copy
   (`scratch/WeatherExt/pkg`, copied from `sims/corrected`, strips installed)
   with `PYTHONPATH` pointing at it, then
   `python -m smart_buildings.run_scenario_generator --buildings building_train_8 --weathers chicago_1`
   generated `training_scenarios/building_train_8/chicago_1` (K=3 zones,
   71 continuous + 4 discrete actions, 283 observations), and a 12-step
   zero-action rollout ran clean: no NaN, no out-of-range, rewards all 0.0.
   Same for `houston_1` (DST-crossing window) and scenario generation for
   `dubai_2`.
2. **Timeline alignment.** At step 0 the observed outside-air temperature was
   272.05 K = -1.1 °C, exactly the `TempC` at `2023-01-08 06:00` in
   `weather/csv/chicago.csv` — i.e. the strip starts where the YAML says in the
   same local frame the CSV is written in.
3. **Boundary behaviour.** `ReplayWeatherController` interpolates through the
   spring-forward label gap without error and raises past the last label
   (2023-12-31 23:00); all strips start early enough that their 30-day window
   ends by 2023-12-12.
4. **Convention check against the shipped CSVs**
   (`python sbx/weather_ext.py --validate-shipped`, same stations re-derived
   from Meteostat): annual mean agrees within 1.1 K (fairbanks -0.9/-0.9,
   phoenix 24.6/25.0, atlanta 17.9/18.6, denver 11.4/10.4, minneapolis
   8.9/10.1, singapore 28.0/28.2 °C) and the diurnal extremes land in the same
   local hours (denver and phoenix exactly; others within 2 h). Day-to-day
   agreement is only moderate (rho 0.53–0.90, MAE 1.5–6.2 K): the shipped
   strips come from a different upstream feed, so they cannot be reproduced
   bit-exactly and are not supposed to be. What this pins down is the units,
   the Celsius/Fahrenheit handling and the local-time convention.
5. **No simulator bugs used.** Nothing here depends on the boiler billing
   sawtooth, return-air area weighting or the fixed occupancy RNG seed; the
   strips are plain recorded weather.

## How to use

```bash
# regenerate everything (downloads are cached in raw/)
./starter-kit/.venv/bin/python sbx/weather_ext.py
./starter-kit/.venv/bin/python sbx/weather_ext.py --refresh --stations dubai

# install the strips into a simulator working copy (adds 51 yamls + 17 csvs)
./starter-kit/.venv/bin/python sbx/weather_ext.py \
    --copy-to sims/corrected/smart_buildings/weather

# generate scenarios and smoke-test one (one process at a time)
cd sims/corrected
PYTHONPATH=$PWD ../../starter-kit/.venv/bin/python -m smart_buildings.run_scenario_generator \
    --buildings building_train_8 --weathers chicago_1
PYTHONPATH=$PWD ../../starter-kit/.venv/bin/python -m smart_buildings.run_environment \
    --building building_train_8 --weather chicago_1 --steps 20
```

## Known limitations

* One station per climate, one year (2023), one provider. Treat differences
  between strips as climate signal, not measurement noise.
* Local-time stamps with DST mean one absent hour per spring transition; the
  simulator interpolates across it, which is exactly what it does for the
  shipped strips.
* Meteostat occasionally carries model-influenced values for sparse periods;
  the shipped comparability check above bounds the resulting error to ~1 K in
  the annual mean.
