"""
Pull daily solar/wind/temperature history for Algerian locations from the
NASA POWER API (free, no key required) and cache both raw JSON and a tidy
combined CSV.

Usage:
    python src/fetch_power_data.py --start 20040101 --end 20231231
"""
import argparse
import csv
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCATIONS_CSV = ROOT / "data" / "locations.csv"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# ALLSKY_SFC_SW_DWN: global horizontal irradiance (GHI) — flat-panel PV yield
# CLRSKY_SFC_SW_DWN: clear-sky GHI — lets us compute a "clearness index" per day
# ALLSKY_SFC_SW_DNI: direct normal irradiance — relevant for CSP / tracking PV
# WS10M / WS50M: wind speed at 10m and 50m (turbine hub-height proxy)
# T2M: air temperature — PV panel efficiency drops as temperature rises
# RH2M: relative humidity
# PRECTOTCORR: precipitation — soiling / cloud-cover context
PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",
    "CLRSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DNI",
    "WS10M",
    "WS50M",
    "T2M",
    "RH2M",
    "PRECTOTCORR",
]


def load_locations():
    with open(LOCATIONS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fetch_location(name, lat, lon, start, end, retries=3):
    params = ",".join(PARAMETERS)
    url = (
        f"{API_URL}?parameters={params}&community=RE"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start}&end={end}&format=JSON"
    )
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to fetch {name} after {retries} attempts: {last_err}")


def flatten_to_rows(name, wilaya, zone, payload):
    param_data = payload["properties"]["parameter"]
    dates = sorted(next(iter(param_data.values())).keys())
    rows = []
    for date in dates:
        row = {"name": name, "wilaya": wilaya, "zone": zone, "date": date}
        for p in PARAMETERS:
            val = param_data[p].get(date)
            row[p] = None if val == -999.0 else val
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20040101")
    parser.add_argument("--end", default="20231231")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    locations = load_locations()
    all_rows = []

    for loc in locations:
        name = loc["name"]
        raw_path = RAW_DIR / f"{name.replace(' ', '_')}.json"

        if raw_path.exists():
            print(f"[skip fetch] {name} — cached at {raw_path}")
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            print(f"[fetch] {name} ({loc['lat']}, {loc['lon']})...")
            payload = fetch_location(name, loc["lat"], loc["lon"], args.start, args.end)
            raw_path.write_text(json.dumps(payload), encoding="utf-8")
            time.sleep(1)  # be polite to the API

        all_rows.extend(flatten_to_rows(name, loc["wilaya"], loc["zone"], payload))

    out_path = PROCESSED_DIR / "power_daily_algeria.csv"
    fieldnames = ["name", "wilaya", "zone", "date"] + PARAMETERS
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows for {len(locations)} locations -> {out_path}")


if __name__ == "__main__":
    main()
