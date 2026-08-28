"""
Re-runs the clustering and forecasting logic from notebooks 02 and 03
and saves lightweight CSV exports for the Streamlit app (app.py) to
load. The app itself never touches sklearn/statsmodels/tensorflow; it
only reads these precomputed files, so it stays fast and has no
retraining non-determinism.

Usage:
    python src/export_app_data.py
"""
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Clustering (mirrors notebooks/02_clustering.ipynb)
# ---------------------------------------------------------------------------
print("Clustering...")
locations = pd.read_csv(ROOT / "data" / "locations.csv")
summary = pd.read_csv(PROCESSED / "site_summary_features.csv", index_col=0)

CLUSTER_FEATURES = [
    "ALLSKY_SFC_SW_DWN", "WS50M", "mean_kt",
    "interannual_cv_pct", "solar_wind_monthly_corr", "hot_days_per_year",
]
X_scaled = StandardScaler().fit_transform(summary[CLUSTER_FEATURES])

hier = AgglomerativeClustering(n_clusters=4, linkage="ward").fit(X_scaled)
summary["cluster"] = hier.labels_

pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
summary["pc1"], summary["pc2"] = coords[:, 0], coords[:, 1]

CLUSTER_NAMES = {
    0: "Coast & Plateau (hybrid hedge)",
    1: "Deep Sahara (high resource & aligned)",
    2: "Saharan Atlas transition",
    3: "Adrar (heat-risk outlier)",
}
summary["cluster_name"] = summary["cluster"].map(CLUSTER_NAMES)

site_clusters = summary.reset_index().rename(columns={"index": "name"})
site_clusters.to_csv(PROCESSED / "site_clusters.csv", index=False)
print(f"  wrote {PROCESSED / 'site_clusters.csv'}")

profile = summary.groupby("cluster")[CLUSTER_FEATURES].mean().round(3)
profile["n_sites"] = summary.groupby("cluster").size()
profile["sites"] = summary.groupby("cluster").apply(lambda g: ", ".join(g.index), include_groups=False)
profile["cluster_name"] = profile.index.map(CLUSTER_NAMES)
profile.reset_index().to_csv(PROCESSED / "cluster_profiles.csv", index=False)
print(f"  wrote {PROCESSED / 'cluster_profiles.csv'}")

# ---------------------------------------------------------------------------
# Forecasting (mirrors notebooks/03_forecasting.ipynb)
# ---------------------------------------------------------------------------
print("Forecasting...")
df = pd.read_csv(PROCESSED / "power_daily_algeria.csv", parse_dates=["date"])
df["kt"] = df["ALLSKY_SFC_SW_DWN"] / df["CLRSKY_SFC_SW_DWN"]

SITES = ["Algiers", "Ouargla"]
TRAIN_END = "2019-12-31"


def build_features(site_df):
    d = site_df.sort_values("date").reset_index(drop=True).copy()
    d["doy"] = d["date"].dt.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * d["doy"] / 366)
    d["doy_cos"] = np.cos(2 * np.pi * d["doy"] / 366)
    for lag in [1, 2, 3, 7]:
        d[f"kt_lag{lag}"] = d["kt"].shift(lag)
    d["kt_roll7_mean"] = d["kt"].shift(1).rolling(7).mean()
    d["kt_roll30_mean"] = d["kt"].shift(1).rolling(30).mean()
    train_mask = d["date"] <= TRAIN_END
    clim = d.loc[train_mask].groupby("doy")["kt"].mean().rename("kt_climatology")
    d = d.merge(clim, on="doy", how="left")
    return d.dropna().reset_index(drop=True)


FEATURE_COLS = [
    "kt_lag1", "kt_lag2", "kt_lag3", "kt_lag7",
    "kt_roll7_mean", "kt_roll30_mean",
    "doy_sin", "doy_cos", "kt_climatology",
]

site_data = {site: build_features(df[df["name"] == site]) for site in SITES}

metrics_rows = []
all_preds = {}
for site in SITES:
    d = site_data[site].copy()
    d["anomaly"] = d["kt"] - d["kt_climatology"]
    d["anomaly_lag1"] = d["anomaly"].shift(1)
    d = d.dropna()

    train = d[d["date"] <= TRAIN_END]
    test = d[d["date"] > TRAIN_END]

    phi = np.corrcoef(train["anomaly_lag1"], train["anomaly"])[0, 1] * (
        train["anomaly"].std() / train["anomaly_lag1"].std()
    )
    ar1_pred = test["kt_climatology"] + phi * test["anomaly_lag1"]

    X_train, y_train = train[FEATURE_COLS], train["kt"]
    X_test, y_test = test[FEATURE_COLS], test["kt"]

    gbr = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0)
    gbr.fit(X_train, y_train)
    gbr_pred = gbr.predict(X_test)

    persistence_pred = test["kt_lag1"]
    climatology_pred = test["kt_climatology"]

    preds = pd.DataFrame({
        "date": test["date"].values,
        "actual": y_test.values,
        "persistence": persistence_pred.values,
        "climatology": climatology_pred.values,
        "ar1": ar1_pred.values,
        "gbr": gbr_pred,
    })

    for label, pred_col in [
        ("Persistence", "persistence"), ("Climatology", "climatology"),
        ("Seasonal-naive + AR(1)", "ar1"), ("Gradient Boosting", "gbr"),
    ]:
        metrics_rows.append({
            "site": site, "method": label,
            "MAE": mean_absolute_error(preds["actual"], preds[pred_col]),
            "RMSE": np.sqrt(mean_squared_error(preds["actual"], preds[pred_col])),
            "R2": r2_score(preds["actual"], preds[pred_col]),
        })

    all_preds[site] = preds

print("Training LSTM for Algiers (one run, cached for the app)...")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.random.set_seed(0)
np.random.seed(0)

WINDOW = 30
d = site_data["Algiers"]
kt_vals = d["kt"].values
aux_vals = d[["doy_sin", "doy_cos", "kt_climatology"]].values
dates = d["date"].values

X_seq, X_aux, y, seq_dates = [], [], [], []
for i in range(WINDOW, len(d)):
    X_seq.append(kt_vals[i - WINDOW:i])
    X_aux.append(aux_vals[i])
    y.append(kt_vals[i])
    seq_dates.append(dates[i])

X_seq = np.array(X_seq)[..., None]
X_aux = np.array(X_aux)
y = np.array(y)
seq_dates = pd.to_datetime(seq_dates)

train_mask = seq_dates <= "2017-12-31"
val_mask = (seq_dates > "2017-12-31") & (seq_dates <= TRAIN_END)
test_mask = seq_dates > TRAIN_END

seq_input = keras.Input(shape=(WINDOW, 1))
x = layers.LSTM(32)(seq_input)
aux_input = keras.Input(shape=(3,))
combined = layers.concatenate([x, aux_input])
combined = layers.Dense(16, activation="relu")(combined)
output = layers.Dense(1)(combined)
lstm_model = keras.Model([seq_input, aux_input], output)
lstm_model.compile(optimizer="adam", loss="mse")
lstm_model.fit(
    [X_seq[train_mask], X_aux[train_mask]], y[train_mask],
    validation_data=([X_seq[val_mask], X_aux[val_mask]], y[val_mask]),
    epochs=100, batch_size=32, verbose=0,
    callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
)
lstm_pred = lstm_model.predict([X_seq[test_mask], X_aux[test_mask]], verbose=0).flatten()

lstm_dates = seq_dates[test_mask]
all_preds["Algiers"] = all_preds["Algiers"].merge(
    pd.DataFrame({"date": lstm_dates, "lstm": lstm_pred}), on="date", how="left"
)

metrics_rows.append({
    "site": "Algiers", "method": "LSTM",
    "MAE": mean_absolute_error(y[test_mask], lstm_pred),
    "RMSE": np.sqrt(mean_squared_error(y[test_mask], lstm_pred)),
    "R2": r2_score(y[test_mask], lstm_pred),
})

all_preds["Algiers"].to_csv(PROCESSED / "forecast_predictions_algiers.csv", index=False)
print(f"  wrote {PROCESSED / 'forecast_predictions_algiers.csv'}")

all_preds["Ouargla"].to_csv(PROCESSED / "forecast_predictions_ouargla.csv", index=False)
print(f"  wrote {PROCESSED / 'forecast_predictions_ouargla.csv'}")

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(PROCESSED / "forecast_metrics.csv", index=False)
print(f"  wrote {PROCESSED / 'forecast_metrics.csv'}")

print("\nDone.")
