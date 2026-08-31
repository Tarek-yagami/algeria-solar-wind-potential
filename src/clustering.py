"""
Shared clustering configuration for the site-typology analysis.

Imported by notebooks/02_clustering.ipynb (exploration/model selection)
and src/export_app_data.py (regenerates the app's CSVs), so both use the
exact same feature set and cluster labels.
"""

CLUSTER_FEATURES = [
    "ALLSKY_SFC_SW_DWN",        # resource strength
    "WS50M",                    # wind resource strength
    "mean_kt",                  # predictability (clearness)
    "interannual_cv_pct",       # year-to-year stability
    "solar_wind_monthly_corr",  # hybrid hedging value
    "hot_days_per_year",        # PV heat-derating risk
]

N_CLUSTERS = 4

CLUSTER_NAMES = {
    0: "Coast & Plateau (hybrid hedge)",
    1: "Deep Sahara (high resource & aligned)",
    2: "Saharan Atlas transition",
    3: "Adrar (heat-risk outlier)",
}
