# Where Should Algeria Build Solar and Wind Capacity?

A resource-assessment and profiling study of renewable energy potential
across Algeria, using 20 years of satellite-derived climate data.

## Problem Statement

Algeria has announced a target of ~15,000 MW of solar capacity by 2035
(the "Tafouk 1" program) and sits on some of the best solar resource in the
world, yet solar still supplies well under 1% of national electricity.
Site selection for that build-out isn't just "where is it sunniest": it
depends on how *predictable* the resource is, how it varies seasonally,
and whether pairing sites with complementary profiles (e.g. a volatile
solar region with a steadier wind corridor) could reduce grid balancing
risk.

**Question:** Using historical solar irradiance, wind speed, and
temperature data for a spread of Algerian locations (coast → high plateau
→ Sahara), can we (1) forecast short-term resource variability, (2)
quantify what drives that variability, and (3) group locations into
distinct renewable profiles that are useful for site-selection reasoning?

This deliberately does not require Algeria's grid operator (Sonelgaz) to
publish anything. We model the *resource*, which is public satellite
data, not the internal grid.

## Data

- **Source:** [NASA POWER API](https://power.larc.nasa.gov/): free,
  no API key, MERRA-2/SYN1deg-derived daily climate data, 1981–present.
- **Locations:** 13 sites spanning Algeria's climate zones: Tell/coastal
  (Algiers, Oran, Annaba), High Plateau (Tiaret, Djelfa, Batna), Saharan
  Atlas (El Bayadh, Laghouat), and Sahara proper, north to deep south
  (Ghardaïa, Ouargla, Béchar, Adrar, Tamanrasset). See
  [data/locations.csv](data/locations.csv).
- **Variables:** all-sky and clear-sky GHI, DNI, wind speed at 10m/50m,
  temperature, relative humidity, precipitation. Chosen because they
  drive *both* PV yield and turbine output, and because clear-sky vs.
  all-sky GHI gives a direct "clearness index" (a measurable proxy for
  cloud-driven unpredictability).
- **Range:** 2004–2023 (20 years), daily resolution. Enough to separate
  genuine seasonal/interannual variability from noise.

Fetched via [src/fetch_power_data.py](src/fetch_power_data.py); raw
per-location JSON is cached in `data/raw/`, combined tidy data lands in
`data/processed/power_daily_algeria.csv`.

## Methodology

1. **EDA & feature engineering**: clearness index (all-sky/clear-sky
   GHI), degree-day-style temperature aggregates, seasonal decomposition
   per site.
2. **Regression**: predict daily GHI/DNI and wind yield from
   engineered features; quantify what actually drives variability at
   each site (feature importance).
3. **Time series forecasting**: baseline (seasonal-naive), then classical
   ML (gradient boosting on lag/rolling features), then LSTM/GRU *only if it
   meaningfully beats classical ML* on short-term (cloud-driven) solar
   variability. The comparison itself is a finding, not a foregone
   conclusion.
4. **Clustering**: group the 13 locations (or site-months) into
   renewable profiles, e.g. high-solar/low-variability,
   high-solar/high-variability, wind-viable-complement. Check whether
   clusters recover the real climate-zone structure or reveal something
   non-obvious.
5. **Interpretability & write-up**: tie findings back to the siting
   question. Which regions look most "bankable" for solar, where would
   wind meaningfully de-risk a hybrid site, and what are this analysis's
   limits (satellite-derived data vs. ground-truth pyranometer
   measurements, no grid-connection or land-use constraints modeled).

## Setup / Reproduce

```bash
pip install -r requirements.txt

# 1. Pull the 20-year climate dataset (takes a few minutes, hits the
#    free NASA POWER API, no key needed)
python src/fetch_power_data.py --start 20040101 --end 20231231

# 2. Run the notebooks in order (each reads the previous step's output)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_clustering.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_forecasting.ipynb
```

Or just open the `.ipynb` files in Jupyter/VS Code and run them
top to bottom; the repo already ships the executed versions with all
outputs and figures, so this step is only needed to regenerate results
from scratch.

## Interactive Dashboard

A Streamlit app for exploring the results without reading through
notebook cells: a map of the 13 sites colored by cluster with a
per-site detail panel, a cluster-profile comparison view, and a
forecasting tab (MAE by method, actual-vs-forecast by year, for both
Algiers and Ouargla).

```bash
pip install -r requirements.txt
python src/export_app_data.py   # only needed once, to (re)generate data/processed/*.csv for the app
streamlit run app.py
```

The app only reads precomputed CSVs, produced by
[src/export_app_data.py](src/export_app_data.py). That script shares
its feature engineering and model code with notebooks 02 and 03 (via
[src/clustering.py](src/clustering.py) and
[src/forecasting.py](src/forecasting.py)), so the app never retrains
anything live and its numbers can't drift from what the notebooks and
write-up report.

## Repo structure

```
data/
  locations.csv          # the 13 sites and their coordinates/climate zone
  raw/                   # cached raw API JSON per location (gitignored)
  processed/             # tidy combined CSV used for analysis + app data
notebooks/               # EDA, clustering, forecasting notebooks (executed)
src/
  fetch_power_data.py    # data acquisition script
  clustering.py          # shared clustering config (notebook 02 + export script)
  forecasting.py         # shared feature engineering & model defs (notebook 03 + export script)
  export_app_data.py     # regenerates the CSVs app.py reads
reports/                 # final write-up / figures
app.py                   # Streamlit dashboard
requirements.txt         # pinned dependencies
```

## Status

- [x] Data acquisition: 20 years daily, 13 sites, zero missing values
- [x] EDA: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)
- [x] Clustering: [notebooks/02_clustering.ipynb](notebooks/02_clustering.ipynb)
- [x] Forecasting: [notebooks/03_forecasting.ipynb](notebooks/03_forecasting.ipynb)
- [x] Write-up: [reports/findings_report.md](reports/findings_report.md)

### Headline finding

Solar/wind seasonal complementarity flips sign moving south: strongly
negative on the coast (wind compensates for weak-solar months, a real
hybrid-plant hedge) to strongly positive in the Sahara (wind and solar
peak together, so a hybrid plant there adds capacity, not risk
diversification). Forecasting independently confirms the same
variability split the clustering step found: real forecast skill at
variable coastal sites, none at stable desert sites, and an LSTM that
does not beat Gradient Boosting once tested fairly.

Full findings, the four-cluster site typology, and limitations:
**[reports/findings_report.md](reports/findings_report.md)**.
