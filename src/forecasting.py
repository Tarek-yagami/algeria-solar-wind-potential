"""
Shared feature engineering and model definitions for the clearness-index
forecasting pipeline.

Imported by notebooks/03_forecasting.ipynb (exploration/model comparison)
and src/export_app_data.py (regenerates the app's CSVs), so both train
and evaluate the exact same models.
"""
import numpy as np

SITES = ["Algiers", "Ouargla"]
TRAIN_END = "2019-12-31"

FEATURE_COLS = [
    "kt_lag1", "kt_lag2", "kt_lag3", "kt_lag7",
    "kt_roll7_mean", "kt_roll30_mean",
    "doy_sin", "doy_cos", "kt_climatology",
]

GBR_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0)

LSTM_WINDOW = 30


def build_features(site_df, train_end=TRAIN_END):
    d = site_df.sort_values("date").reset_index(drop=True).copy()
    d["doy"] = d["date"].dt.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * d["doy"] / 366)
    d["doy_cos"] = np.cos(2 * np.pi * d["doy"] / 366)

    for lag in [1, 2, 3, 7]:
        d[f"kt_lag{lag}"] = d["kt"].shift(lag)

    d["kt_roll7_mean"] = d["kt"].shift(1).rolling(7).mean()
    d["kt_roll30_mean"] = d["kt"].shift(1).rolling(30).mean()

    train_mask = d["date"] <= train_end
    clim = d.loc[train_mask].groupby("doy")["kt"].mean().rename("kt_climatology")
    d = d.merge(clim, on="doy", how="left")

    return d.dropna().reset_index(drop=True)


def fit_ar1(train, test):
    """Seasonal-naive + AR(1) on the climatology anomaly."""
    phi = np.corrcoef(train["anomaly_lag1"], train["anomaly"])[0, 1] * (
        train["anomaly"].std() / train["anomaly_lag1"].std()
    )
    ar1_pred = test["kt_climatology"] + phi * test["anomaly_lag1"]
    return ar1_pred, phi


def build_lstm_model(window, n_aux):
    from tensorflow import keras
    from tensorflow.keras import layers

    seq_input = keras.Input(shape=(window, 1), name="kt_sequence")
    x = layers.LSTM(32)(seq_input)
    aux_input = keras.Input(shape=(n_aux,), name="calendar_climatology")
    combined = layers.concatenate([x, aux_input])
    combined = layers.Dense(16, activation="relu")(combined)
    output = layers.Dense(1)(combined)
    model = keras.Model(inputs=[seq_input, aux_input], outputs=output)
    model.compile(optimizer="adam", loss="mse")
    return model
