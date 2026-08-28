# %% [markdown]
# # EDA: Solar & Wind Resource Across Algeria's Climate Zones
#
# Question this notebook starts to answer: **do the 13 sites actually
# separate into meaningfully different renewable profiles, and what drives
# the difference?** Before any modeling, we need to see whether "coast vs.
# plateau vs. Sahara" is a real signal in the data or just a label we
# imposed.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
FIG_DIR = "../reports/figures"
import os
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv("../data/processed/power_daily_algeria.csv", parse_dates=["date"])
locations = pd.read_csv("../data/locations.csv")

df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

# order sites north -> south (the real geographic gradient), derived from
# the data rather than hardcoded
site_order = locations.sort_values("lat", ascending=False)["name"].tolist()
zone_order = (
    locations.sort_values("lat", ascending=False)
    .drop_duplicates("zone")["zone"]
    .tolist()
)
palette = dict(zip(zone_order, sns.color_palette("colorblind", len(zone_order))))

print(df.shape)
df.head()

# %% [markdown]
# ## 1. Clearness index: how predictable is the solar resource?
#
# `ALLSKY_SFC_SW_DWN / CLRSKY_SFC_SW_DWN` measures how much cloud cover
# actually knocks down irradiance relative to a cloudless sky. A site
# stuck at ~0.55 has structurally cloudy weather; a site at ~0.75+ is
# close to its physical ceiling most days, which matters more for
# "bankability" than raw average irradiance alone.

# %%
df["kt"] = df["ALLSKY_SFC_SW_DWN"] / df["CLRSKY_SFC_SW_DWN"]

kt_stats = (
    df.groupby("name")["kt"]
    .agg(mean_kt="mean", std_kt="std")
    .reindex(site_order)
)
kt_stats["zone"] = locations.set_index("name").loc[kt_stats.index, "zone"]
kt_stats

# %%
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(
    kt_stats.index,
    kt_stats["mean_kt"],
    yerr=kt_stats["std_kt"],
    color=[palette[z] for z in kt_stats["zone"]],
    capsize=3,
)
ax.set_ylabel("Clearness index (mean ± std)")
ax.set_title("Clearness index by site, north → south")
plt.xticks(rotation=45, ha="right")
handles = [plt.Rectangle((0, 0), 1, 1, color=palette[z]) for z in zone_order]
ax.legend(handles, zone_order, title="Zone", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/clearness_index_by_site.png", dpi=150)
plt.show()

# %% [markdown]
# ## 2. Solar & wind resource ranking
#
# Raw annual-mean GHI/DNI tells us "how much energy is physically
# available"; wind speed at 50m is a rough turbine hub-height proxy.

# %%
annual_means = (
    df.groupby("name")[["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "WS10M", "WS50M", "T2M"]]
    .mean()
    .reindex(site_order)
)
annual_means["zone"] = locations.set_index("name").loc[annual_means.index, "zone"]
annual_means

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col, title, ylabel in zip(
    axes,
    ["ALLSKY_SFC_SW_DWN", "WS50M"],
    ["Mean daily GHI", "Mean wind speed at 50m"],
    ["kWh/m²/day", "m/s"],
):
    ax.bar(
        annual_means.index,
        annual_means[col],
        color=[palette[z] for z in annual_means["zone"]],
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/solar_wind_ranking.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Seasonal profiles by zone
#
# Averaging sites within each hand-labeled zone, month by month. This is
# the first check on whether "zone" is doing real work or is just a label.

# %%
monthly_by_zone = (
    df.merge(locations[["name", "zone"]], on="name", suffixes=("", "_loc"))
    .groupby(["zone", "month"])[["ALLSKY_SFC_SW_DWN", "WS50M", "T2M"]]
    .mean()
    .reset_index()
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, col, title, ylabel in zip(
    axes,
    ["ALLSKY_SFC_SW_DWN", "WS50M", "T2M"],
    ["GHI", "Wind speed (50m)", "Temperature"],
    ["kWh/m²/day", "m/s", "°C"],
):
    for zone in zone_order:
        sub = monthly_by_zone[monthly_by_zone["zone"] == zone]
        ax.plot(sub["month"], sub[col], marker="o", label=zone, color=palette[zone])
    ax.set_title(f"Monthly mean {title} by zone")
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, 13))

axes[0].legend(fontsize=8, loc="lower center")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/seasonal_profiles_by_zone.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Interannual variability: is a good year repeatable?
#
# Average clearness index per site *per year*, then look at the spread
# across the 20 years. A site with a high mean but wide year-to-year
# spread is a riskier bet for a 25-year asset than a site with a slightly
# lower but rock-steady mean.

# %%
annual_kt = df.groupby(["name", "year"])["kt"].mean().reset_index()
interannual_spread = (
    annual_kt.groupby("name")["kt"]
    .agg(mean_of_years="mean", std_across_years="std")
    .reindex(site_order)
)
interannual_spread["cv_pct"] = (
    100 * interannual_spread["std_across_years"] / interannual_spread["mean_of_years"]
)
interannual_spread

# %%
fig, ax = plt.subplots(figsize=(9, 6))
zones_for_sites = locations.set_index("name").loc[interannual_spread.index, "zone"]
ax.scatter(
    interannual_spread["mean_of_years"],
    interannual_spread["cv_pct"],
    c=[palette[z] for z in zones_for_sites],
    s=80,
)
for name, row in interannual_spread.iterrows():
    ax.annotate(name, (row["mean_of_years"], row["cv_pct"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("Mean clearness index (20-yr average)")
ax.set_ylabel("Interannual variability (coefficient of variation, %)")
ax.set_title("Resource strength vs. year-to-year stability")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/interannual_stability.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Solar/wind complementarity
#
# For each site, does wind tend to be *higher* when solar is *lower*
# (a natural hedge for a hybrid plant), or do they move together (no
# hedging benefit, possibly compounding risk)? Using monthly climatology
# per site to see the seasonal phase relationship, not day-to-day noise.

# %%
monthly_by_site = (
    df.groupby(["name", "month"])[["ALLSKY_SFC_SW_DWN", "WS50M"]].mean().reset_index()
)
complementarity = (
    monthly_by_site.groupby("name")
    .apply(lambda g: g["ALLSKY_SFC_SW_DWN"].corr(g["WS50M"]))
    .reindex(site_order)
    .rename("solar_wind_monthly_corr")
)
complementarity_df = complementarity.to_frame()
complementarity_df["zone"] = locations.set_index("name").loc[complementarity_df.index, "zone"]
complementarity_df.sort_values("solar_wind_monthly_corr")

# %% [markdown]
# Negative correlation = wind climatologically compensates for weaker
# solar months at that site (a real hybrid-plant argument). Positive
# correlation = the two resources are seasonally aligned, so pairing them
# doesn't reduce seasonal variability, it just adds more capacity in the
# same good/bad months.

# %%
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#c0392b" if v > 0 else "#2471a3" for v in complementarity_df["solar_wind_monthly_corr"]]
ax.barh(complementarity_df.index, complementarity_df["solar_wind_monthly_corr"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Monthly solar–wind correlation (climatology)")
ax.set_title("Solar/wind seasonal complementarity by site")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/solar_wind_complementarity.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Heat exposure: a PV-derating signal
#
# PV panel efficiency drops as cell temperature rises above ~25°C.
# Counting days over 35°C air temperature is a rough proxy for which
# sites will see real-world yield below their nameplate irradiance would
# suggest.

# %%
hot_days = (
    df[df["T2M"] > 35]
    .groupby("name")
    .size()
    .reindex(site_order, fill_value=0)
    .rename("days_over_35C_per_20yr")
    .to_frame()
)
hot_days["days_per_year"] = hot_days["days_over_35C_per_20yr"] / 20
hot_days

# %% [markdown]
# ## Summary table for the next stage (clustering)
#
# One row per site with the features that will feed the clustering step:
# resource strength, predictability, interannual stability, solar/wind
# complementarity, and heat exposure.

# %%
summary = (
    annual_means[["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "WS10M", "WS50M", "T2M"]]
    .join(kt_stats[["mean_kt", "std_kt"]])
    .join(interannual_spread[["cv_pct"]].rename(columns={"cv_pct": "interannual_cv_pct"}))
    .join(complementarity_df[["solar_wind_monthly_corr"]])
    .join(hot_days[["days_per_year"]].rename(columns={"days_per_year": "hot_days_per_year"}))
)
summary["zone"] = locations.set_index("name").loc[summary.index, "zone"]
summary["lat"] = locations.set_index("name").loc[summary.index, "lat"]
summary["lon"] = locations.set_index("name").loc[summary.index, "lon"]

summary.to_csv("../data/processed/site_summary_features.csv")
summary

# %% [markdown]
# ## Key findings so far
#
# 1. **The zone labels are real, not decorative.** Clearness index rises
#    almost monotonically from ~0.80–0.85 on the coast to ~0.93–0.95 deep
#    in the Sahara, and interannual variability drops from ~2%+ CV on the
#    coast/plateau to under 1% in the desert. The Sahara isn't just
#    sunnier, it's also structurally more *predictable* year over year,
#    which matters as much as raw resource strength for a 25-year asset.
#
# 2. **Solar/wind complementarity flips sign north to south.** Coastal and
#    plateau sites show strong *negative* solar–wind correlation
#    (Annaba −0.86, Tiaret −0.83, Algiers −0.74): wind climatologically
#    compensates for weaker solar months, a real hybrid-plant hedging
#    argument. Deep-Sahara sites show the opposite: *positive*
#    correlation (Béchar +0.80, Ouargla +0.73, Adrar +0.57): wind and
#    solar peak in the same months there, so a hybrid plant adds capacity
#    without reducing seasonal variability. This is a genuinely
#    non-obvious result and directly informs where a hybrid vs.
#    single-resource plant makes sense.
#
# 3. **Raw solar ranking is not the same as usable solar ranking.** Adrar
#    has the 2nd-highest GHI of all 13 sites but also 87 days/year above
#    35°C, enough heat exposure to meaningfully derate real-world PV
#    yield (panel efficiency drops with cell temperature). Tamanrasset,
#    the single highest-GHI site, sees *zero* days over 35°C in this
#    dataset, despite being further south, a coastal-vs-continental /
#    altitude effect worth digging into rather than assuming "further
#    south = hotter."
#
# These three points are exactly what the clustering stage (next
# notebook) needs to test formally: do sites group along resource
# strength, or does variability/complementarity/heat produce a different,
# more decision-relevant typology?
