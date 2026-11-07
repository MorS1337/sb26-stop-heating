#!/usr/bin/env python3
"""Build an external weather set for climate-transfer validation.

The graded Kaggle run uses weather stations that never appear in the shipped
training scenarios (shipped: fairbanks, denver, atlanta, phoenix, singapore).
This module downloads 2023 hourly observations for a diverse set of *unseen*
stations, converts them into the simulator's weather CSV schema, and emits one
drop-in weather-preset directory that can be copied into any working copy of
``smart_buildings/weather/``.

Source
------
Meteostat v2 bulk hourly endpoint (free, no API key, CC BY 4.0 for the data,
code MIT): https://bulk.meteostat.net/v2/hourly/<station>.csv.gz
Meteostat aggregates the national weather services' SYNOP/METAR reports
(mostly NOAA ISD + national agencies). Columns of the bulk file are
``date,hour,temp,dwpt,rhum,prcp,snow,wdir,wspd,wpgt,pres,tsun,coco`` where
``date``/``hour`` are **UTC**, ``temp``/``dwpt`` in degC, ``rhum`` in percent,
``wspd`` in km/h.

Simulator contract (verified against smart_buildings/smart_control/simulator/
weather_controller.py: read_weather_csv)
------------------------------------------------------------------------
* The CSV must carry a ``Time`` column formatted ``%Y%m%d-%H%M``; pandas parses
  it with ``utc=True``, so whatever we write is interpreted as UTC, and
  ``np.interp`` runs over seconds-since-epoch.
* Only ``TempF`` and ``Humidity`` are actually consumed (``TempC``,
  ``DewPointC``, ``WindspeedKmph``, ``StationName``, ``Location`` are carried
  for humans/plotting). ``TempF`` is interpolated, humidity is interpolated.
* The shipped CSVs were built from local station clock time (their diurnal
  minimum sits at 03:00-06:00 and the maximum at 13:00-15:00, which is only
  possible if the stamps are local, not UTC, even though the loader calls them
  UTC). So we convert UTC -> local station time here: this keeps the diurnal
  cycle aligned to the hour-of-day features the policy observes, exactly like
  the shipped strips. DST: labels are true local wall clock (tz database), a
  spring-forward hour is absent, the duplicated fall-back hour is averaged, so
  2023 yields 8759 or 8760 rows.
* A preset YAML carries ``name``, ``weather_csv`` (path relative to the
  package weather/ dir), ``convection_coefficient`` and ``start_timestamp``.
  ``start_timestamp`` is the *episode* start and must lie inside the CSV; a
  strip is a 30 day / 8640 step window starting Sunday 06:00.

convection_coefficient convention
---------------------------------
The eight strips shipped per station share one constant coefficient per
station (fairbanks 8.8, singapore 17.6, atlanta 19.1, phoenix 19.5,
minneapolis 21.5, denver 26.7 W/m2/K); the repo contains no generator that
derives it. Over the six shipped stations the value tracks annual mean wind
speed v [m/s] as h = 1.72 + 5.32*v (least squares over the shipped CSV wind
columns, RMSE 2.0, max error 2.3 W/m2/K). We use exactly that fit, clipped to
the shipped range [8.0, 27.0] and rounded to 0.1, so our stations sit on the
same manifold as the shipped ones instead of inventing a new convention.

Usage
-----
    # fetch + convert everything, write CSVs, strip YAMLs, report.json
    python sbx/weather_ext.py

    # re-download even if the raw cache exists
    python sbx/weather_ext.py --refresh

    # copy the drop-in tree into a simulator working copy
    python sbx/weather_ext.py --copy-to sims/corrected/smart_buildings/weather

    # check the conversion against the five shipped CSVs (sanity, not a test)
    python sbx/weather_ext.py --validate-shipped

Outputs (all under data/weather_ext/, nothing is ever written into starter-kit/):
    raw/<id>.csv.gz            raw Meteostat bulk file (cache)
    weather/csv/<slug>.csv     simulator-format weather CSV
    weather/<slug>_<n>.yaml    drop-in preset (3 strips per station)
    report.json                gaps, stats, convection coefficients, strips
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import ssl
import sys
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

YEAR = 2023
BULK_URL = "https://bulk.meteostat.net/v2/hourly/{station_id}.csv.gz"
MAX_GAP = 0.05  # drop a station whose 2023 temperature gap exceeds 5%
MAX_STRIP_STEPS = 8640  # 30 days at 5 min

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "weather_ext")
RAW_DIR = os.path.join(OUT_DIR, "raw")
WEATHER_DIR = os.path.join(OUT_DIR, "weather")
CSV_DIR = os.path.join(WEATHER_DIR, "csv")

# --- convection coefficient fit, see module docstring -----------------------
H_INTERCEPT = 1.72
H_SLOPE = 5.32
H_CLIP = (8.0, 27.0)


@dataclass
class Station:
  """One external weather station and how we slice it into strips."""

  station_id: str  # Meteostat station id (bulk endpoint / station metadata)
  slug: str  # our file stem
  label: str  # StationName column, dotted like the shipped CSVs
  location: str  # Location column
  tz: str  # IANA timezone of the station (local clock time)
  climate: str  # short tag, documentation only
  months: tuple[int, ...]  # one strip per month, Sunday 06:00 start
  source_name: str = ""  # human readable Meteostat name
  convection: float = 0.0  # filled in by convert_station()


STATIONS: list[Station] = [
    # --- cold / continental ------------------------------------------------
    Station("72530", "chicago", "Chicago.O'Hare", "Chicago O'Hare, USA",
            "America/Chicago", "continental-humid", (1, 7, 10),
            "Chicago O'Hare Airport"),
    Station("70273", "anchorage", "Anchorage.Intl.AP", "Anchorage, USA",
            "America/Anchorage", "subarctic", (1, 7, 10),
            "Anchorage Ted Stevens Airport"),
    Station("71627", "montreal", "Montreal.Intl.AP", "Montreal, Canada",
            "America/Toronto", "continental-cold", (1, 7, 10),
            "Montreal International Airport"),
    Station("02974", "helsinki", "Helsinki.Vantaa", "Helsinki, Finland",
            "Europe/Helsinki", "nordic", (1, 7, 10), "Helsinki-Vantaa"),
    Station("UUEE0", "moscow", "Moscow.Sheremetyevo", "Moscow, Russia",
            "Europe/Moscow", "severe-continental", (1, 7, 10),
            "Moscow Sheremetyevo"),
    Station("54511", "beijing", "Beijing.Capital", "Beijing, China",
            "Asia/Shanghai", "continental-monsoon", (1, 7, 10),
            "Beijing"),
    Station("04018", "reykjavik", "Reykjavik.Keflavik", "Reykjavik, Iceland",
            "Atlantic/Reykjavik", "subpolar-marine", (1, 7, 10),
            "Keflavikurflugvollur"),
    # --- hot arid ----------------------------------------------------------
    Station("72386", "lasvegas", "Las.Vegas.McCarran", "Las Vegas, USA",
            "America/Los_Angeles", "hot-arid", (4, 7, 9),
            "McCarran International Airport"),
    Station("41194", "dubai", "Dubai.Intl.AP", "Dubai, UAE",
            "Asia/Dubai", "hot-arid-coastal", (1, 7, 9),
            "Dubai International Airport"),
    # --- hot humid ---------------------------------------------------------
    Station("72243", "houston", "Houston.Intercontinental", "Houston, USA",
            "America/Chicago", "hot-humid", (2, 7, 9),
            "Houston Intercontinental"),
    Station("72202", "miami", "Miami.Intl.AP", "Miami, USA",
            "America/New_York", "subtropical-humid", (1, 4, 7),
            "Miami International Airport"),
    Station("48456", "bangkok", "Bangkok.Don.Muang", "Bangkok, Thailand",
            "Asia/Bangkok", "tropical-monsoon", (4, 9, 11),
            "Don Muang"),
    Station("47671", "tokyo", "Tokyo.Haneda", "Tokyo, Japan",
            "Asia/Tokyo", "humid-subtropical", (1, 8, 10),
            "Tokyo International Airport"),
    # --- mild marine -------------------------------------------------------
    Station("72793", "seattle", "Seattle.Tacoma", "Seattle, USA",
            "America/Los_Angeles", "marine-mild", (1, 7, 10),
            "Seattle-Tacoma Airport"),
    Station("72494", "sanfrancisco", "San.Francisco.Intl.AP",
            "San Francisco, USA", "America/Los_Angeles", "cool-marine", (1, 7, 9),
            "San Francisco Airport"),
    Station("03772", "london", "London.Heathrow", "London, UK",
            "Europe/London", "oceanic-temperate", (1, 7, 10),
            "London Heathrow Airport"),
    # --- southern hemisphere (seasonal inversion) --------------------------
    Station("93119", "auckland", "Auckland.Intl.AP", "Auckland, New Zealand",
            "Pacific/Auckland", "southern-marine", (1, 7, 10),
            "Auckland Airport"),
]

# Shipped stations we can re-derive from Meteostat, for a convention check:
# slug -> (Meteostat station id, IANA timezone).
SHIPPED_REMAP = {
    "atlanta": ("72219", "America/New_York"),
    "denver": ("72565", "America/Denver"),
    "fairbanks": ("70261", "America/Anchorage"),
    "minneapolis": ("72658", "America/Chicago"),
    "phoenix": ("72278", "America/Phoenix"),
    "singapore": ("48698", "Asia/Singapore"),
}
STARTER_KIT_WEATHER = os.path.join(
    ROOT, "starter-kit", "smart_buildings", "weather", "csv"
)


# --- download --------------------------------------------------------------
def raw_path(station_id: str) -> str:
  return os.path.join(RAW_DIR, f"{station_id}.csv.gz")


def _ssl_context() -> ssl.SSLContext:
  """Verified TLS context; the venv python lacks a system CA bundle."""
  try:
    import certifi  # noqa: PLC0415  (optional: present in starter-kit venv)

    return ssl.create_default_context(cafile=certifi.where())
  except ImportError:
    return ssl.create_default_context()


def download(station_id: str, refresh: bool = False) -> str:
  """Download (or reuse) the Meteostat bulk file for one station."""
  path = raw_path(station_id)
  if os.path.exists(path) and not refresh:
    return path
  os.makedirs(RAW_DIR, exist_ok=True)
  url = BULK_URL.format(station_id=station_id)
  request = urllib.request.Request(url, headers={"User-Agent": "sbx-weather-ext"})
  with urllib.request.urlopen(  # noqa: S310
      request, timeout=120, context=_ssl_context()
  ) as response:
    payload = response.read()
  gzip.decompress(payload)  # fail loudly on a truncated/corrupt download
  with open(path, "wb") as handle:
    handle.write(payload)
  return path


# --- conversion ------------------------------------------------------------
def load_station_frame(station: Station, year: int = YEAR) -> pd.DataFrame:
  """Load one station's hourly 2023 observations on a gap-free UTC grid.

  Returns a frame indexed by UTC timestamps with columns
  ``temp, dwpt, rhum, wspd`` and a boolean ``missing`` mask of the raw gaps
  (before interpolation), so gap statistics are measured on real data rather
  than on what we filled in.
  """
  with gzip.open(raw_path(station.station_id), "rt") as handle:
    df = pd.read_csv(
        handle,
        header=None,
        names=["date", "hour", "temp", "dwpt", "rhum", "prcp", "snow",
               "wdir", "wspd", "wpgt", "pres", "tsun", "coco"],
        na_values=[""],
    )
  df = df[["date", "hour", "temp", "dwpt", "rhum", "wspd"]].copy()
  df["time"] = pd.to_datetime(df["date"], utc=True) + pd.to_timedelta(
      df["hour"], unit="h"
  )
  df = df.drop(columns=["date", "hour"]).set_index("time").sort_index()

  # One day of margin each side: the local-year window of an eastern station
  # starts up to 14 h before UTC Jan 1 and ends up to 14 h after UTC Dec 31.
  grid = pd.date_range(
      f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h", tz="UTC"
  )
  grid = pd.date_range(grid[0] - pd.Timedelta(days=1),
                       grid[-1] + pd.Timedelta(days=1), freq="h", tz="UTC")
  df = df.reindex(grid)
  df = df[~df.index.duplicated(keep="first")]
  df["missing"] = df["temp"].isna()
  return df


def gap_fraction(df: pd.DataFrame, column: str) -> float:
  return float(df[column].isna().mean())


def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
  """Linear interpolation in time, then edge fill; also clamp to physics."""
  out = df.copy()
  for column in ("temp", "dwpt", "rhum", "wspd"):
    out[column] = out[column].interpolate(method="time", limit_direction="both")
  out["rhum"] = out["rhum"].clip(0.0, 100.0)
  out["wspd"] = out["wspd"].clip(lower=0.0)
  # dewpoint cannot exceed temperature; Meteostat occasionally violates it.
  out["dwpt"] = np.minimum(out["dwpt"], out["temp"])
  return out


def to_local(df: pd.DataFrame, tz: str, year: int = YEAR) -> pd.DataFrame:
  """Map a UTC-indexed frame to local station wall clock (DST aware).

  Trims to the local calendar year, so the CSV starts at 00:00 local January 1
  and ends at 23:00 local December 31 whichever side of UTC the station is on.
  The fall-back duplicate hour is averaged so the ``%Y%m%d-%H%M`` labels stay
  unique; the spring-forward hour simply does not exist (as on a wall clock).
  """
  local = df.tz_convert(tz).tz_localize(None)
  local = local.groupby(level=0).mean()
  local = local[local.index.year == year]
  return local.sort_index()


def write_weather_csv(local: pd.DataFrame, station: Station) -> str:
  """Write the simulator-format CSV; returns the path."""
  os.makedirs(CSV_DIR, exist_ok=True)
  path = os.path.join(CSV_DIR, f"{station.slug}.csv")
  temp_c = np.round(local["temp"].to_numpy(), 1)
  frame = pd.DataFrame({
      "Time": local.index.strftime("%Y%m%d-%H%M"),
      "StationName": station.label,
      "Location": station.location,
      "TempC": temp_c,
      "DewPointC": np.round(local["dwpt"].to_numpy(), 1),
      "WindspeedKmph": np.round(local["wspd"].to_numpy(), 2),
      "Humidity": np.round(local["rhum"].to_numpy(), 1),
      # derived from the rounded Celsius value so the two columns agree exactly
      "TempF": np.round(temp_c * 9.0 / 5.0 + 32.0, 2),
  })
  frame.to_csv(path, index=True)
  return path


def convection_coefficient(mean_wind_mps: float) -> float:
  """Per-station constant, fitted on the six shipped stations."""
  value = H_INTERCEPT + H_SLOPE * mean_wind_mps
  return round(float(np.clip(value, *H_CLIP)), 1)


# --- strips ----------------------------------------------------------------
SUNDAY = 6  # pandas weekday index


def sunday_start(year: int, month: int, day: int = 8) -> pd.Timestamp:
  """Sunday 06:00 local on/after ``day`` of ``month`` (episode start)."""
  start = pd.Timestamp(year=year, month=month, day=day) + pd.Timedelta(hours=6)
  while start.weekday() != SUNDAY:
    start += pd.Timedelta(days=1)
  return start


def write_strip(station: Station, index: int, start: pd.Timestamp) -> dict:
  """Write one preset YAML; return its summary record."""
  name = f"{station.slug}_{index}"
  path = os.path.join(WEATHER_DIR, f"{name}.yaml")
  text = (
      f"name: {name}\n"
      f"weather_csv: csv/{station.slug}.csv\n"
      f"convection_coefficient: {station.convection:g}\n"
      f"start_timestamp: {start.strftime('%Y-%m-%d %H:%M')}\n"
  )
  with open(path, "w") as handle:
    handle.write(text)
  return {"name": name, "path": os.path.relpath(path, ROOT),
          "start_timestamp": start.strftime("%Y-%m-%d %H:%M")}


def strip_window_stats(local: pd.DataFrame, start: pd.Timestamp,
                       steps: int = MAX_STRIP_STEPS) -> dict:
  """Statistics over the exact 30 day window the episode would replay."""
  end = start + pd.Timedelta(minutes=5 * steps)
  window = local.loc[(local.index >= start) & (local.index < end)]
  if window.empty:
    return {"covers_window": False}
  temp_c = window["temp"]
  return {
      "covers_window": bool(
          window.index.max() >= end - pd.Timedelta(hours=1)
      ),
      "rows": int(len(window)),
      "temp_c_mean": round(float(temp_c.mean()), 2),
      "temp_c_min": round(float(temp_c.min()), 2),
      "temp_c_max": round(float(temp_c.max()), 2),
      "temp_k_min": round(float(temp_c.min()) + 273.15, 2),
      "temp_k_max": round(float(temp_c.max()) + 273.15, 2),
      "rhum_mean": round(float(window["rhum"].mean()), 1),
  }


# --- driver ----------------------------------------------------------------
def convert_station(station: Station, year: int = YEAR) -> dict:
  """Fetch, convert and emit one station. Returns its report record."""
  download(station.station_id)
  raw = load_station_frame(station, year=year)
  # Gaps are measured on the local-year window, i.e. exactly the hours the
  # emitted CSV covers, before any interpolation.
  local_raw = to_local(raw, station.tz, year)
  gaps = {
      "temp": round(gap_fraction(local_raw, "temp"), 4),
      "dwpt": round(gap_fraction(local_raw, "dwpt"), 4),
      "rhum": round(gap_fraction(local_raw, "rhum"), 4),
      "wspd": round(gap_fraction(local_raw, "wspd"), 4),
  }
  station.convection = convection_coefficient(
      float(local_raw["wspd"].mean() / 3.6)  # km/h -> m/s
  )
  record = {
      "slug": station.slug,
      "station_id": station.station_id,
      "source": BULK_URL.format(station_id=station.station_id),
      "name": station.source_name,
      "station_name_column": station.label,
      "location": station.location,
      "tz": station.tz,
      "climate": station.climate,
      "year": year,
      "gaps": gaps,
      "mean_wind_mps": round(float(local_raw["wspd"].mean() / 3.6), 2),
      "convection_coefficient": station.convection,
      "dropped": bool(gaps["temp"] > MAX_GAP),
      "strips": [],
  }
  if record["dropped"]:
    return record

  filled = fill_gaps(raw)
  local = to_local(filled, station.tz, year)
  # Hard invariants: the simulator rejects nothing but a NaN propagates into
  # every downstream value, so fail here rather than inside a rollout.
  assert not local[["temp", "dwpt", "rhum", "wspd"]].isna().any().any(), (
      station.slug
  )
  record["csv"] = os.path.relpath(write_weather_csv(local, station), ROOT)
  record["rows"] = int(len(local))
  record["temp_c_mean"] = round(float(local["temp"].mean()), 2)
  record["temp_c_min"] = round(float(local["temp"].min()), 2)
  record["temp_c_max"] = round(float(local["temp"].max()), 2)

  for index, month in enumerate(station.months, start=1):
    start = sunday_start(year, month)
    strip = write_strip(station, index, start)
    strip.update(strip_window_stats(local, start))
    record["strips"].append(strip)
  return record


def validate_shipped() -> None:
  """Re-derive the shipped stations and compare them to the shipped CSVs.

  Two robust checks: (a) the annual mean agrees within about a kelvin, which
  pins the column scaling and the Celsius/Fahrenheit conversion, and (b) the
  diurnal temperature extremes land in the same clock hours as the shipped
  CSV, which pins the local-time (not UTC) convention. Day-to-day agreement is
  only approximate: the shipped strips come from a different upstream feed
  (Weather Underground style station pages), not from Meteostat.
  """
  for slug, (station_id, tz) in SHIPPED_REMAP.items():
    station = Station(station_id, slug, slug.upper(), slug, tz, "shipped", ())
    download(station_id)
    raw = fill_gaps(load_station_frame(station, year=YEAR))
    shipped_path = os.path.join(STARTER_KIT_WEATHER, f"{slug}.csv")
    if not os.path.exists(shipped_path):
      print(f"{slug:12s} shipped csv missing, skipped")
      continue
    shipped = pd.read_csv(shipped_path)
    shipped.index = pd.to_datetime(shipped["Time"], format="%Y%m%d-%H%M")
    ours = to_local(raw, station.tz, YEAR)["temp"]
    joined = pd.concat(
        [ours.rename("ours"), shipped["TempC"].rename("shipped")], axis=1
    ).dropna()
    scores = {}
    for lag in (-2, -1, 0, 1, 2):
      shifted = joined["ours"].shift(lag)
      mask = shifted.notna() & joined["shipped"].notna()
      scores[lag] = round(
          float(np.corrcoef(shifted[mask], joined["shipped"][mask])[0, 1]), 4
      )
    mae = float((joined["ours"] - joined["shipped"]).abs().mean())
    best = max(scores, key=lambda k: scores[k])
    # Annual mean and diurnal phase are the robust checks: the shipped CSVs
    # come from a different provider, so day to day they only roughly agree.
    ours_mean = float(joined["ours"].mean())
    shipped_mean = float(joined["shipped"].mean())
    ours_hour = joined.groupby(joined.index.hour)["ours"].mean()
    shipped_hour = joined.groupby(joined.index.hour)["shipped"].mean()
    print(
        f"{slug:12s} n={len(joined):5d} annual={shipped_mean:6.2f}/{ours_mean:6.2f}C "
        f"(d={shipped_mean - ours_mean:+.2f}) "
        f"diurnal_shipped={shipped_hour.idxmin()}h/{shipped_hour.idxmax()}h "
        f"diurnal_ours={ours_hour.idxmin()}h/{ours_hour.idxmax()}h "
        f"corr={scores[0]:.3f} mae={mae:5.2f}K best_lag={best:+d}h"
    )


def copy_to(destination: str) -> None:
  """Copy the drop-in weather tree (yaml + csv/) into a simulator weather dir."""
  destination = os.path.abspath(destination)
  os.makedirs(os.path.join(destination, "csv"), exist_ok=True)
  copied = 0
  for name in sorted(os.listdir(WEATHER_DIR)):
    source = os.path.join(WEATHER_DIR, name)
    if name.endswith(".yaml"):
      shutil.copy2(source, os.path.join(destination, name))
      copied += 1
  for name in sorted(os.listdir(CSV_DIR)):
    shutil.copy2(
        os.path.join(CSV_DIR, name), os.path.join(destination, "csv", name)
    )
  print(f"copied {copied} presets + csv/ into {destination}")


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--refresh", action="store_true",
                      help="re-download raw files even if cached")
  parser.add_argument("--stations", nargs="*", default=None,
                      help="subset of slugs (default: all)")
  parser.add_argument("--copy-to", default=None,
                      help="copy presets+csv into a simulator weather/ dir")
  parser.add_argument("--validate-shipped", action="store_true",
                      help="convention check against the shipped CSVs")
  args = parser.parse_args(argv)

  os.makedirs(WEATHER_DIR, exist_ok=True)
  if args.copy_to:
    copy_to(args.copy_to)
    return 0
  if args.validate_shipped:
    validate_shipped()
    return 0

  wanted = set(args.stations) if args.stations else None
  records = []
  for station in STATIONS:
    if wanted and station.slug not in wanted:
      continue
    if args.refresh:
      download(station.station_id, refresh=True)
    record = convert_station(station)
    status = "DROP" if record["dropped"] else "ok"
    strips = len(record["strips"])
    print(
        f"{record['slug']:14s} {status:4s} temps gap={record['gaps']['temp']:.3f}"
        f" rhum gap={record['gaps']['rhum']:.3f}"
        f" v={record['mean_wind_mps']:4.1f}m/s h={record['convection_coefficient']:5.1f}"
        f" strips={strips}"
        + (f" T=[{record['temp_c_min']:.0f},{record['temp_c_max']:.0f}]C"
           if not record["dropped"] else "")
    )
    records.append(record)

  # A subset run (--stations) must not clobber the stations it did not touch.
  merged: dict[str, dict] = {}
  report_path = os.path.join(OUT_DIR, "report.json")
  if wanted and os.path.exists(report_path):
    with open(report_path) as handle:
      for previous in json.load(handle).get("stations", []):
        merged[previous["slug"]] = previous
  for record in records:
    merged[record["slug"]] = record
  records = [merged[key] for key in sorted(merged)] if wanted else records

  report = {
      "generated_by": "sbx/weather_ext.py",
      "source": "Meteostat v2 bulk hourly (https://bulk.meteostat.net/v2/hourly/)",
      "license": "Meteostat data CC BY 4.0; station observations via NOAA ISD "
                 "and national weather services",
      "year": YEAR,
      "max_gap": MAX_GAP,
      "convection_fit": {
          "form": "h = a + b * mean_wind_mps, clipped to [8, 27]",
          "a": H_INTERCEPT,
          "b": H_SLOPE,
          "fitted_on": {k: v for k, v in SHIPPED_REMAP.items()},
      },
      "stations": records,
      "summary": {
          "stations_converted": sum(not r["dropped"] for r in records),
          "stations_dropped": [r["slug"] for r in records if r["dropped"]],
          "strips": sum(len(r["strips"]) for r in records),
      },
  }
  with open(report_path, "w") as handle:
    json.dump(report, handle, indent=2)
  print(json.dumps(report["summary"], indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
