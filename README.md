# Where Should Algeria Build Solar and Wind Capacity?

A resource-assessment and profiling study of renewable energy potential
across Algeria, using 20 years of satellite-derived climate data.

## Problematique

Algeria has announced a target of ~15,000 MW of solar capacity by 2035
(the "Tafouk 1" program) and sits on some of the best solar resource in the
world — yet solar still supplies well under 1% of national electricity.
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
publish anything — we model the *resource*, which is public satellite
data, not the internal grid.

## Data

- **Source:** [NASA POWER API](https://power.larc.nasa.gov/) — free,
  no API key, MERRA-2/SYN1deg-derived daily climate data, 1981–present.
- **Locations:** 13 sites spanning Algeria's climate zones — Tell/coastal
  (Algiers, Oran, Annaba), High Plateau (Tiaret, Djelfa, Batna), Saharan
  Atlas (El Bayadh, Laghouat), and Sahara proper, north to deep south
  (Ghardaïa, Ouargla, Béchar, Adrar, Tamanrasset). See
  [data/locations.csv](data/locations.csv).
- **Variables:** all-sky and clear-sky GHI, DNI, wind speed at 10m/50m,
  temperature, relative humidity, precipitation. Chosen because they
  drive *both* PV yield and turbine output, and because clear-sky vs.
  all-sky GHI gives a direct "clearness index" — a measurable proxy for
  cloud-driven unpredictability.
- **Range:** 2004–2023 (20 years), daily resolution. Enough to separate
  genuine seasonal/interannual variability from noise.

Fetched via [src/fetch_power_data.py](src/fetch_power_data.py); raw
per-location JSON is cached in `data/raw/`, combined tidy data lands in
`data/processed/power_daily_algeria.csv`.

## Planned methodology

1. **EDA & feature engineering** — clearness index (all-sky/clear-sky
   GHI), degree-day-style temperature aggregates, seasonal decomposition
   per site.
2. **Regression** — predict daily GHI/DNI and wind yield from
   engineered features; quantify what actually drives variability at
   each site (feature importance).
3. **Time series forecasting** — baseline (seasonal-naive) → classical
   ML (gradient boosting on lag/rolling features) → LSTM/GRU *only if it
   meaningfully beats classical ML* on short-term (cloud-driven) solar
   variability. The comparison itself is a finding, not a foregone
   conclusion.
4. **Clustering** — group the 13 locations (or site-months) into
   renewable profiles: e.g. high-solar/low-variability,
   high-solar/high-variability, wind-viable-complement. Check whether
   clusters recover the real climate-zone structure or reveal something
   non-obvious.
5. **Interpretability & write-up** — tie findings back to the siting
   question: which regions look most "bankable" for solar, where would
   wind meaningfully de-risk a hybrid site, and what are this analysis's
   limits (satellite-derived data vs. ground-truth pyranometer
   measurements, no grid-connection or land-use constraints modeled).

## Repo structure

```
data/
  locations.csv          # the 13 sites and their coordinates/climate zone
  raw/                   # cached raw API JSON per location
  processed/             # tidy combined CSV used for analysis
notebooks/               # EDA, modeling, clustering notebooks
src/
  fetch_power_data.py    # data acquisition script
reports/                 # final write-up / figures
```

## Status

- [x] Data acquisition — 20 years daily, 13 sites, zero missing values
- [x] EDA — [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)
- [ ] Clustering — site typology
- [ ] Regression / forecasting
- [ ] Write-up

### Key EDA findings

- **Clearness index rises and stabilizes south**: ~0.80–0.85 on the coast
  vs. ~0.93–0.95 in the deep Sahara, with interannual variability
  dropping from 2%+ CV to under 1% — the south isn't just sunnier, it's
  more predictable year over year.
- **Solar/wind complementarity flips sign north to south** (see
  [reports/figures/solar_wind_complementarity.png](reports/figures/solar_wind_complementarity.png)):
  strongly negative on the coast/plateau (wind compensates for weak
  solar months — a real hybrid-plant hedge) to strongly positive in the
  Sahara (wind and solar peak together — a hybrid plant there adds
  capacity, not risk diversification).
- **Raw solar ranking ≠ usable solar ranking**: Adrar has the 2nd-highest
  GHI of all 13 sites but 87 days/year above 35°C, enough to meaningfully
  derate real PV yield — while Tamanrasset, the single highest-GHI site,
  sees zero days over 35°C in this dataset.
