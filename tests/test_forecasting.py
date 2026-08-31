import numpy as np
import pandas as pd
import pytest

from src.forecasting import FEATURE_COLS, build_features, fit_ar1

# Spans a leap year (2016) inside the training window so every day-of-year
# (1-366) has a climatology value before the 2020 test period is reached.
FULL_RANGE = pd.date_range("2016-01-01", "2020-12-31", freq="D")
TRAIN_END = "2019-12-31"


def make_site_df():
    kt = 0.5 + 0.1 * np.sin(2 * np.pi * FULL_RANGE.dayofyear / 366)
    return pd.DataFrame({"date": FULL_RANGE, "kt": kt})


def test_build_features_has_all_feature_columns_with_no_missing_values():
    result = build_features(make_site_df(), train_end=TRAIN_END)
    for col in FEATURE_COLS:
        assert col in result.columns
        assert result[col].notna().all()


def test_build_features_only_drops_rows_missing_rolling_history():
    df = make_site_df()
    result = build_features(df, train_end=TRAIN_END)
    # kt_roll30_mean needs 30 prior days; that's the only thing that should
    # cost us rows once every day-of-year is covered by the training window.
    assert len(result) == len(df) - 30
    assert result["date"].iloc[0] == df["date"].iloc[30]


def test_build_features_climatology_does_not_leak_from_test_period():
    df = make_site_df()
    df.loc[df["date"] > TRAIN_END, "kt"] = 999.0

    result = build_features(df, train_end=TRAIN_END)

    assert result["kt_climatology"].max() < 10


def test_fit_ar1_recovers_known_slope():
    anomaly = pd.Series(np.linspace(-1, 1, 20))
    train = pd.DataFrame({"anomaly_lag1": anomaly, "anomaly": 2 * anomaly})
    test = pd.DataFrame({"kt_climatology": [0.5, 0.6], "anomaly_lag1": [0.1, -0.1]})

    ar1_pred, phi = fit_ar1(train, test)

    assert phi == pytest.approx(2.0)
    assert list(ar1_pred) == pytest.approx([0.5 + 2 * 0.1, 0.6 + 2 * -0.1])
