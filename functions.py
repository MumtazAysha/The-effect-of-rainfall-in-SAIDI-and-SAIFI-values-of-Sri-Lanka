# functions.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# Validation
# ---------------------------
def validate_data(df):
    required = ["Month", "SAIDI", "SAIFI", "Rainfall"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required)

    # Convert Month names (e.g. "July") to numbers (7)
    try:
        df["Month_Num"] = pd.to_datetime(df["Month"], format="%B").dt.month
    except Exception:
        # If Month column already numeric
        df["Month_Num"] = pd.to_numeric(df["Month"], errors="coerce")

    df = df.dropna(subset=["Month_Num"])
    df["Month_Num"] = df["Month_Num"].astype(int)

    return df


def run_correlations(x, y, metric, label="Overall"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 3:
        # Not enough points for a meaningful correlation — return NaNs but keep a row
        return {
            "Scope": label, "Metric": metric, "N": n,
            "Pearson_r": np.nan, "Pearson_p": np.nan,
            "Spearman_r": np.nan, "Spearman_p": np.nan
        }

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    print(f"\n📊 {metric} correlations for {label}:")
    print(f" Pearson r = {pearson_r:.3f}, p = {pearson_p:.3f}")
    print(f" Spearman rho = {spearman_r:.3f}, p = {spearman_p:.3f}")

    return {
        "Scope": label, "Metric": metric, "N": n,
        "Pearson_r": pearson_r, "Pearson_p": pearson_p,
        "Spearman_r": spearman_r, "Spearman_p": spearman_p
    }

def run_regression(x, y, metric, label="Overall", make_plots=False):
    import matplotlib.pyplot as plt
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 2:
        # Not enough points to fit a line — return NaNs but keep a row
        return {
            "Scope": label, "Metric": metric, "N": n,
            "Slope": np.nan, "Intercept": np.nan,
            "R2": np.nan, "RMSE": np.nan
        }

    X = x.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    r2 = float(r2_score(y, y_pred))

    print(f"\n📈 {metric} Regression for {label}:")
    print(f" Slope = {model.coef_[0]:.3f}, Intercept = {model.intercept_:.3f}")
    print(f" R² = {r2:.3f}, RMSE = {rmse:.3f}")

    if make_plots:
        order = np.argsort(x)
        plt.scatter(x, y, label="Actual data")
        plt.plot(x[order], y_pred[order], color="red", label="Regression line")
        plt.xlabel("Rainfall (mm)")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Rainfall — {label}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "Scope": label, "Metric": metric, "N": n,
        "Slope": float(model.coef_[0]), "Intercept": float(model.intercept_),
        "R2": r2, "RMSE": rmse
    }

def predict_future(df_future, pickle_file="saidi_saifi_multiyear.pkl"):
    """
    Predict SAIDI and SAIFI for future years using trained multi-year models.
    Expects df_future with columns: Year, Month_Num, Rainfall
    """
    import pickle
    with open(pickle_file, "rb") as f:
        bundle = pickle.load(f)

    model_saidi = bundle["model_saidi"]
    model_saifi = bundle["model_saifi"]

    df_future = df_future.copy()
    df_future["SAIDI_pred"] = model_saidi.predict(df_future[["Year", "Month_Num", "Rainfall"]])
    df_future["SAIFI_pred"] = model_saifi.predict(df_future[["Year", "Month_Num", "Rainfall"]])

    return df_future
