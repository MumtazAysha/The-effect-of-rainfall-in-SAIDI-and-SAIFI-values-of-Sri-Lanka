# makepickle.py
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

DATA_PATH = "Average national rainfall and SAIDI-SAIFI_2024.csv"
PICKLE_OUT = "saidi_saifi_model.pkl"

def validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    need = ["Month", "SAIDI", "SAIFI", "Rainfall"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    df = df.dropna(subset=need).copy()
    # Month → Month_Num
    try:
        df["Month_Num"] = pd.to_datetime(df["Month"], format="%B").dt.month
    except Exception:
        df["Month_Num"] = pd.to_numeric(df["Month"], errors="coerce")
    df = df.dropna(subset=["Month_Num"])
    df["Month_Num"] = df["Month_Num"].astype(int)
    return df

def main():
    df = pd.read_csv(DATA_PATH)
    df = validate_and_prepare(df)

    # --- Full models: [Month_Num, Rainfall] -> metric
    models_full = {}
    for metric in ["SAIDI", "SAIFI"]:
        X = df[["Month_Num", "Rainfall"]].to_numpy(dtype=float)
        y = df[metric].to_numpy(dtype=float)
        m = LinearRegression().fit(X, y)
        models_full[metric] = m

    # --- Month-only backup models: [Month_Num] -> metric
    models_month_only = {}
    for metric in ["SAIDI", "SAIFI"]:
        X = df[["Month_Num"]].to_numpy(dtype=float)
        y = df[metric].to_numpy(dtype=float)
        m = LinearRegression().fit(X, y)
        models_month_only[metric] = m

    # --- Baseline rainfall by month (from 2024; used when rainfall is unknown)
    baseline = (
        df.groupby("Month_Num")["Rainfall"]
          .mean()
          .reindex(range(1,13))
          .fillna(method="ffill")
          .fillna(method="bfill")
          .to_dict()
    )

    bundle = {
        "full": models_full,               # expects features [Month_Num, Rainfall]
        "month_only": models_month_only,   # expects features [Month_Num] only
        "baseline_rainfall_by_month": baseline,
        "meta": {
            "data_source": DATA_PATH,
            "features_full": ["Month_Num", "Rainfall"],
            "features_month_only": ["Month_Num"]
        }
    }

    with open(PICKLE_OUT, "wb") as f:
        pickle.dump(bundle, f)

    print(f"✅ Saved models to {PICKLE_OUT}")
    print("   - full: [Month_Num, Rainfall] -> SAIDI/SAIFI")
    print("   - month_only: [Month_Num] -> SAIDI/SAIFI (fallback)")
    print("   - baseline_rainfall_by_month included")

if __name__ == "__main__":
    main()
