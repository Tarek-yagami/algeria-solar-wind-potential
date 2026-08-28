# Where Should Algeria Build Solar and Wind Capacity?

**Findings report: synthesis of `01_eda`, `02_clustering`, `03_forecasting`**

## Problem Statement

Algeria has announced a ~15,000 MW solar target (Tafouk 1) and sits on
some of the best solar resource in the world, yet solar supplies well
under 1% of national electricity today. Site selection for that
build-out isn't just "where is it sunniest": it depends on how
*predictable* the resource is, how it varies seasonally, and whether
pairing solar with wind at a given site actually reduces risk or just
adds capacity in the same good/bad months.

Using 20 years of daily satellite-derived climate data (NASA POWER,
2004–2023) across 13 sites spanning Algeria's climate zones (Tell/coastal,
High Plateau, Saharan Atlas, and Sahara proper), this project asked
whether the data supports a more decision-relevant answer than
"the Sahara is sunny."

## What the data showed

### 1. Resource strength and predictability both rise moving south, and together

Clearness index (actual GHI ÷ theoretical cloud-free GHI) climbs from
~0.80–0.85 on the coast to ~0.93–0.95 in the deep Sahara, while
interannual variability drops from 2%+ (coefficient of variation across
20 years) to under 1%. The Sahara isn't just sunnier, it's also
structurally *more repeatable* year over year, which matters as much as
raw resource strength when underwriting a 25-year asset.

### 2. Solar/wind seasonal complementarity flips sign, north to south

This is the project's centerpiece finding. Correlating each site's
monthly solar and wind climatology shows a near-monotonic sign flip:

| Region | Solar–wind monthly correlation | Reading |
|---|---|---|
| Annaba, Tiaret, Algiers, Oran (coast/plateau) | −0.86 to −0.74 | Wind climatologically compensates for weak-solar months, a real hybrid-plant hedge |
| Djelfa, Laghouat, El Bayadh (transition) | −0.45 to −0.21 | Partial hedge |
| Ghardaïa | +0.06 | Hinge point |
| Ouargla, Béchar, Tamanrasset, Adrar (deep Sahara) | +0.40 to +0.80 | Wind and solar peak *together*, so pairing adds capacity, not risk reduction |

A hybrid solar+wind plant is a genuinely different proposition on the
coast than in the Sahara, and a resource map alone (which only shows
"how much energy," not "when") would never surface this.

### 3. Raw solar ranking is not the same as usable solar ranking

Adrar has the 2nd-highest GHI of all 13 sites but also 87 days/year
above 35°C, enough heat exposure to meaningfully derate real-world PV
yield (panel efficiency drops as cell temperature rises).
Tamanrasset, the single highest-GHI site, sees zero days over 35°C in
this dataset despite being further south: a reminder that "further
into the Sahara" isn't a reliable proxy for "hotter," and that raw
irradiance rankings need a heat-exposure check before they inform capital
decisions.

### 4. A data-driven clustering mostly, but not entirely, recovers the hand-drawn zones

Clustering purely on resource strength, predictability, complementarity,
and heat exposure (no location information) produced four groups:

- **Coast & Plateau, hybrid hedge** (Annaba, Algiers, Oran, Batna,
  Tiaret): moderate resource, weakest year-to-year stability of the
  four, but by far the strongest solar/wind hedge (−0.74).
- **Deep Sahara, high resource & aligned** (Ghardaïa, Ouargla, Béchar,
  Tamanrasset): strong, stable resource, but solar and wind now move
  together, the best case for solar-only builds, not hybrid risk reduction.
- **Saharan Atlas transition** (Djelfa, Laghouat, El Bayadh): resource
  and stability sit between the other two groups, but this cluster has
  the **highest mean wind speed of any non-outlier group** (6.35 m/s at
  50m), a genuinely non-obvious result that makes it worth a dedicated
  wind-resource look, not a leftover "in-between" label.
- **Adrar, heat-risk outlier** (singleton): the single best raw
  resource of all 13 sites, undercut by heat exposure 5x any other
  cluster.

The one real mismatch with hand-drawn geography: **Djelfa**, labeled
"High Plateau" like Batna and Tiaret, actually clusters with the Saharan
Atlas sites. Its climate behavior doesn't match the administrative/
geographic label it was given.

### 5. Classical time series diagnostics confirm the structure ML found on its own

STL decomposition (trend/seasonal/residual) plus ACF/PACF on the
residual shows a textbook **AR(1) signature** at both sites: PACF cuts
off sharply after lag 1 (0.32→0.06 at Algiers, 0.23→0.02 at Ouargla)
while ACF decays gradually. That independently reproduces what the
Gradient Boosting permutation importance already found (`kt_lag1`
dominant, every other lag/rolling feature marginal), a classical
statistical method and an ML model's internal feature ranking agreeing
on the same structure. The decomposition also re-surfaces the coastal/
Sahara variability gap directly: Algiers' residual is ~2x noisier than
Ouargla's (std 0.176 vs. 0.089).

### 6. Forecasting independently confirms the variability split, and shows where it does and doesn't matter

Running a full model ladder, persistence, then climatology, then a classical
**seasonal-naive + AR(1)** model (deseasonalized with train-only
climatology, per the diagnostics above), then Gradient Boosting, then LSTM, on
one site from each end of the variability spectrum:

- **Algiers** (coastal, high-variability cluster): naive baselines
  score *negative* R², worse than guessing the average. The classical
  AR(1) model already reaches positive R² (0.165), confirming the
  autocorrelation is real, learnable structure, but Gradient Boosting
  still edges it out on both MAE (0.1261 vs. 0.1293) and R² (0.193 vs.
  0.165), and beats climatology by ~15% MAE.
- **Ouargla** (deep Sahara, stable cluster): nothing beats plain
  persistence (MAE 0.0516) in any meaningful way, not Gradient Boosting
  (0.0518), not AR(1) (0.0532). Forecasting earns nothing here, because
  the site is already pinned near its ceiling most days.
- **LSTM vs. Gradient Boosting at Algiers** (the only site with signal
  to chase): across repeated training runs, including with TensorFlow's
  threading pinned for reproducibility, LSTM MAE landed between 0.1220
  and 0.1253 (GBR: 0.1261). It never came out worse than GBR, but the
  exact margin (0.7% to 3.3% MAE improvement) shifted meaningfully
  between runs of the *identical* code purely from TensorFlow's own
  training non-determinism. That's a small, directionally consistent
  edge, but not a precisely quotable one, and it came at the cost of a
  custom two-input Keras architecture and materially more setup and
  runtime than a single `GradientBoostingRegressor.fit()` call. **Given
  the size and instability of the gain, Gradient Boosting is still the
  practical recommendation for this task**, with the LSTM's consistent
  (if variable) edge and the classical AR(1) model's much cheaper
  near-tie both worth keeping in mind if squeezing out the last bit of
  accuracy matters more than simplicity and reproducibility.

This is the same Cluster-0-vs-Cluster-1 variability split the clustering
step found, now confirmed from two independent angles (a classical
statistical model and forecast skill, not just raw variance): multiple
analyses converging on the same structural conclusion.

## Synthesis: what this means for siting

| Cluster | Best case | Caveat |
|---|---|---|
| Coast & Plateau | Hybrid solar+wind, explicitly for the seasonal hedge | Needs real forecasting/curtailment planning: it's the one group where variability is high *and* partially unpredictable |
| Deep Sahara (Ghardaïa/Ouargla/Béchar/Tamanrasset) | Large solar-only builds | A hybrid plant here adds capacity, not risk diversification; don't sell it as a hedge |
| Saharan Atlas transition (Djelfa/Laghouat/El Bayadh) | Underrated wind resource, worth a dedicated look | Currently reads as a "leftover" zone; the data says otherwise |
| Adrar | Best raw resource of all 13 sites | Not a safe pick without a PV-derating / thermal-management study first |

## Limitations

- **Satellite-derived, not ground-truth.** NASA POWER is MERRA-2/SYN1deg
  reanalysis data, reliable and consistent across all 13 sites (which
  is exactly why it was usable at all, given Algeria doesn't publish
  granular grid or ground-station data), but it is not a substitute for
  on-site pyranometer/anemometer measurements before any actual
  investment decision.
- **50m wind speed is a rough hub-height proxy.** Modern utility turbines
  sit closer to 100–120m; actual turbine yield would need
  extrapolation or better data.
- **No grid connection, transmission distance, land use, or cost-of-capital
  modeling.** This is a resource and predictability study, not a full
  site-selection or economic feasibility study; those layers would sit
  on top of these findings, not replace them.
- **13 points, not a continuous map.** The clustering and complementarity
  results describe these specific locations; interpolating between them
  to cover all of Algeria would need a proper spatial model, not a
  straight line on the map.
- **Forecast R² throughout is modest (best case 0.19–0.20).** Day-ahead
  cloud cover has real irreducible uncertainty without actual
  weather-forecast/satellite-nowcasting inputs. This project
  characterizes resource *predictability*, it does not claim a
  production-grade forecasting product.

## Repo

- [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb): clearness index, seasonal profiles, complementarity, heat exposure
- [`notebooks/02_clustering.ipynb`](../notebooks/02_clustering.ipynb): site typology, hand-labeled zones vs. data-driven clusters
- [`notebooks/03_forecasting.ipynb`](../notebooks/03_forecasting.ipynb): baselines, Gradient Boosting, LSTM comparison
- [`data/processed/site_summary_features.csv`](../data/processed/site_summary_features.csv): the per-site feature table used for clustering
