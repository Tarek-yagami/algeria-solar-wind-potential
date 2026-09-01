"""
Interactive companion to the notebooks: explore the 13 sites, the
data-driven cluster typology, and the forecasting results without
reading through cell-by-cell notebook output.

Run locally:
    streamlit run app.py

All data here is precomputed by src/export_app_data.py (which mirrors
the exact logic in notebooks/02_clustering.ipynb and
notebooks/03_forecasting.ipynb) and cached as CSVs in data/processed/.
This app only reads those files; it never retrains anything, so it
stays fast and its numbers never drift from what the notebooks report.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.clustering import CLUSTER_NAMES

st.set_page_config(page_title="Algeria Solar & Wind Siting", layout="wide")

PROCESSED = "data/processed"


@st.cache_data
def load_data():
    return {
        "sites": pd.read_csv(f"{PROCESSED}/site_clusters.csv"),
        "cluster_profiles": pd.read_csv(f"{PROCESSED}/cluster_profiles.csv"),
        "metrics": pd.read_csv(f"{PROCESSED}/forecast_metrics.csv"),
        "algiers": pd.read_csv(f"{PROCESSED}/forecast_predictions_algiers.csv", parse_dates=["date"]),
        "ouargla": pd.read_csv(f"{PROCESSED}/forecast_predictions_ouargla.csv", parse_dates=["date"]),
    }


data = load_data()
sites = data["sites"]
cluster_profiles = data["cluster_profiles"]
metrics = data["metrics"]

CLUSTER_COLOR_PALETTE = ["#2471a3", "#e67e22", "#27ae60", "#c0392b"]
CLUSTER_COLORS = {CLUSTER_NAMES[i]: CLUSTER_COLOR_PALETTE[i] for i in sorted(CLUSTER_NAMES)}

st.title("Where Should Algeria Build Solar and Wind Capacity?")
st.caption(
    "20 years of NASA POWER climate data across 13 sites, interactive companion to the "
    "[full write-up](reports/findings_report.md) and notebooks in this repo."
)

with st.expander("Problem statement", expanded=False):
    st.markdown(
        """
Algeria has announced a ~15,000 MW solar target and sits on some of the
best solar resource in the world, yet solar supplies well under 1% of
national electricity today. Site selection isn't just "where is it
sunniest": it depends on how *predictable* the resource is, how it
varies seasonally, and whether pairing solar with wind at a given site
actually reduces risk or just adds capacity in the same good/bad
months. This explores that question across 13 sites spanning Algeria's
climate zones, coast to deep Sahara.
        """
    )

tab_map, tab_clusters, tab_forecast = st.tabs(
    ["Site Map & Clusters", "Cluster Profiles", "Forecasting"]
)

# ---------------------------------------------------------------------------
with tab_map:
    fig = px.scatter_map(
        sites, lat="lat", lon="lon", color="cluster_name",
        hover_name="name",
        hover_data={
            "lat": False, "lon": False, "cluster_name": False,
            "ALLSKY_SFC_SW_DWN": ":.2f", "mean_kt": ":.2f",
            "solar_wind_monthly_corr": ":.2f", "hot_days_per_year": ":.0f",
        },
        color_discrete_map=CLUSTER_COLORS,
        center={"lat": 29.8, "lon": 2.8}, zoom=4.2, height=560,
        map_style="carto-positron",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="Cluster")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Site detail")
    site_name = st.selectbox("Select a site", sites["name"].tolist())
    row = sites[sites["name"] == site_name].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Mean GHI (kWh/m²/day)", f"{row['ALLSKY_SFC_SW_DWN']:.2f}",
        help="Average daily global horizontal irradiance (all-sky) over the 20-year record.",
    )
    c2.metric(
        "Clearness index", f"{row['mean_kt']:.2f}",
        help="Actual GHI divided by theoretical cloud-free GHI. Higher means clearer, more predictable skies.",
    )
    c3.metric(
        "Wind speed at 50m (m/s)", f"{row['WS50M']:.2f}",
        help="Average wind speed at 50 meters, a proxy for turbine hub height.",
    )
    c4.metric(
        "Interannual variability", f"{row['interannual_cv_pct']:.2f}%",
        help="Coefficient of variation in mean irradiance across years. Higher means less predictable year to year.",
    )
    c5.metric(
        "Hot days/year (>35°C)", f"{row['hot_days_per_year']:.0f}",
        help="Days per year where the *daily mean* temperature (not the daily high) exceeded 35°C.",
    )

    corr = row["solar_wind_monthly_corr"]
    hedge = "hedges seasonal variability" if corr < -0.2 else (
        "adds capacity without reducing seasonal risk" if corr > 0.2 else "roughly neutral"
    )
    st.markdown(
        f"**Cluster:** {row['cluster_name']}  \n"
        f"**Zone (hand-labeled):** {row['zone']}  \n"
        f"**Solar/wind monthly correlation:** {corr:.2f} ({hedge} for a hybrid plant here)"
    )

# ---------------------------------------------------------------------------
with tab_clusters:
    st.subheader("Four data-driven renewable siting profiles")
    st.caption(
        "Clustered purely on resource strength, predictability, solar/wind "
        "complementarity, and heat exposure, with no location information given "
        "to the model."
    )

    for _, prof in cluster_profiles.iterrows():
        with st.container(border=True):
            st.markdown(f"### {prof['cluster_name']}")
            st.caption(f"Sites: {prof['sites']}  ·  n={int(prof['n_sites'])}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Mean GHI", f"{prof['ALLSKY_SFC_SW_DWN']:.2f}",
                help="Average daily global horizontal irradiance (all-sky, kWh/m²/day) over the 20-year record.",
            )
            c2.metric(
                "Clearness index", f"{prof['mean_kt']:.2f}",
                help="Actual GHI divided by theoretical cloud-free GHI. Higher means clearer, more predictable skies.",
            )
            c3.metric(
                "Solar/wind corr", f"{prof['solar_wind_monthly_corr']:.2f}",
                help="Monthly correlation between solar and wind resource. Negative means wind compensates for weak-solar months; positive means they peak together.",
            )
            c4.metric(
                "Hot days/yr", f"{prof['hot_days_per_year']:.0f}",
                help="Days per year where the *daily mean* temperature (not the daily high) exceeded 35°C.",
            )

    col_pca, col_radar = st.columns(2)

    with col_pca:
        st.subheader("Where each site sits in feature space")
        fig_pca = px.scatter(
            sites, x="pc1", y="pc2", color="cluster_name", text="name",
            color_discrete_map=CLUSTER_COLORS,
            hover_data={"pc1": False, "pc2": False, "cluster_name": False, "zone": True},
            height=480,
        )
        fig_pca.update_traces(textposition="top center", marker=dict(size=11))
        fig_pca.update_layout(legend_title_text="", xaxis_title="PC1", yaxis_title="PC2",
                               legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_pca, width="stretch")
        st.caption(
            "Same PCA projection as the clustering notebook, interactive here: "
            "hover a site for its hand-labeled zone, zoom/pan to inspect crowded areas."
        )

    with col_radar:
        st.subheader("Cluster shape comparison")
        radar_features = ["ALLSKY_SFC_SW_DWN", "WS50M", "mean_kt",
                           "interannual_cv_pct", "solar_wind_monthly_corr", "hot_days_per_year"]
        radar_labels = ["Mean GHI", "Wind (50m)", "Clearness",
                        "Interannual CV", "Solar/wind corr", "Hot days/yr"]
        radar_df = cluster_profiles.set_index("cluster_name")[radar_features]
        normalized = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())
        normalized = normalized.fillna(0.5)

        fig_radar = go.Figure()
        for cluster_name in radar_df.index:
            vals = normalized.loc[cluster_name].tolist()
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=radar_labels + [radar_labels[0]],
                fill="toself", name=cluster_name, opacity=0.6,
                line=dict(color=CLUSTER_COLORS.get(cluster_name)),
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
            height=480, legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_radar, width="stretch")
        st.caption(
            "Each axis is min-max normalized across the 4 clusters (not absolute "
            "units), so shapes are directly comparable. Click a cluster in the "
            "legend to isolate it."
        )

    st.subheader("Compare clusters on one feature")
    feature = st.selectbox(
        "Feature",
        ["ALLSKY_SFC_SW_DWN", "WS50M", "mean_kt", "interannual_cv_pct",
         "solar_wind_monthly_corr", "hot_days_per_year"],
        format_func=lambda x: {
            "ALLSKY_SFC_SW_DWN": "Mean GHI (kWh/m²/day)", "WS50M": "Wind speed at 50m (m/s)",
            "mean_kt": "Clearness index", "interannual_cv_pct": "Interannual variability (%)",
            "solar_wind_monthly_corr": "Solar/wind monthly correlation",
            "hot_days_per_year": "Hot days/year (daily mean >35°C)",
        }[x],
    )
    fig_bar = px.bar(
        cluster_profiles, x="cluster_name", y=feature, color="cluster_name",
        color_discrete_map=CLUSTER_COLORS,
    )
    fig_bar.update_layout(showlegend=False, xaxis_title="", height=420)
    st.plotly_chart(fig_bar, width="stretch")

# ---------------------------------------------------------------------------
with tab_forecast:
    st.subheader("Day-ahead clearness index forecasting")
    site_choice = st.radio(
        "Site", ["Algiers", "Ouargla"], horizontal=True,
        help="Algiers: coastal, high-variability cluster. Ouargla: deep Sahara, stable cluster.",
    )
    preds = data["algiers"] if site_choice == "Algiers" else data["ouargla"]
    site_metrics = metrics[metrics["site"] == site_choice].sort_values("MAE")

    if site_choice == "Ouargla":
        st.caption(
            "No LSTM row here, and that's deliberate, not missing data: Gradient "
            "Boosting doesn't beat plain persistence at Ouargla (see the chart "
            "below), so there's no learnable signal for a bigger model to chase. "
            "The LSTM was only trained at Algiers, the one site where Gradient "
            "Boosting actually found something worth improving on."
        )

    method_colors = {
        "Persistence": "#7f8c8d", "Climatology": "#95a5a6",
        "Seasonal-naive + AR(1)": "#e67e22", "Gradient Boosting": "#2471a3",
        "LSTM": "#8e44ad",
    }

    c1, c2 = st.columns([1, 1.4])
    with c1:
        fig_mae = px.bar(
            site_metrics, x="method", y="MAE", color="method",
            color_discrete_map=method_colors,
        )
        fig_mae.update_layout(showlegend=False, xaxis_title="", height=380,
                               title=f"{site_choice}: forecast MAE by method")
        st.plotly_chart(fig_mae, width="stretch")
    with c2:
        st.dataframe(
            site_metrics[["method", "MAE", "RMSE", "R2"]].round(4),
            hide_index=True, width="stretch",
        )

    st.subheader("Actual vs. forecast")
    years = sorted(preds["date"].dt.year.unique())
    year = st.select_slider("Year", options=years, value=2021 if 2021 in years else years[0])
    sub = preds[preds["date"].dt.year == year]

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=sub["date"], y=sub["actual"], name="Actual", line=dict(color="black", width=1)))
    for col, label in [("gbr", "Gradient Boosting"), ("climatology", "Climatology"), ("lstm", "LSTM")]:
        if col in sub.columns and sub[col].notna().any():
            fig_line.add_trace(go.Scatter(
                x=sub["date"], y=sub[col], name=label,
                line=dict(color=method_colors.get(label), dash="dash" if col == "climatology" else None),
                opacity=0.85,
            ))
    fig_line.update_layout(height=420, yaxis_title="kt", title=f"{site_choice}: {year} actual vs. forecast")
    st.plotly_chart(fig_line, width="stretch")

    if site_choice == "Algiers":
        st.info(
            "Naive baselines score negative R² here (worse than guessing the average). "
            "Gradient Boosting and LSTM both find real, if modest, learnable structure. "
            "LSTM's edge over GBR is small and unstable across training runs (never worse, "
            "but the margin varies), so Gradient Boosting is the practical recommendation."
        )
    else:
        st.info(
            "Nothing beats plain persistence in any meaningful way here. Ouargla is already "
            "pinned near its ceiling most days, so there's very little to forecast."
        )

st.divider()
st.caption(
    "Data: [NASA POWER API](https://power.larc.nasa.gov/). "
    "Full analysis: `notebooks/01_eda.ipynb`, `notebooks/02_clustering.ipynb`, "
    "`notebooks/03_forecasting.ipynb`, and `reports/findings_report.md`."
)
