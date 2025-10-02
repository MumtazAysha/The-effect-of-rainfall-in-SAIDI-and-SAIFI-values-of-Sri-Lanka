# build_pickle_from_training.py
# Build a fresh pickle from your training CSV (Rainfall-SAIDI-SAIFI.csv)

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ==== CONFIG: change if your filename is different ====
TRAIN_CSV = "Rainfall-SAIDI-SAIFI.csv"
OUT_PKL   = "saidi_saifi_multiyear.pkl"

REQUIRED_BASE = {"Rainfall", "SAIDI", "SAIFI"}

def fail(msg: str):
    raise SystemExit(f"\nERROR: {msg}\n")

def load_training_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        fail(f"Training CSV not found at: {p.resolve()}")
    df = pd.read_csv(p)
    # normalize column names (strip only; keep original case for display)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def prepare_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Month_Num (1..12) from either:
      - Month (names like January/Jan or numbers 1..12)
      - Month_Num directly
    """
    df = df.copy()
    cols = set(df.columns)

    if "Month_Num" in cols:
        df["Month_Num"] = pd.to_numeric(df["Month_Num"], errors="coerce")
        return df

    if "Month" in cols:
        # try month names like "January"
        try:
            mnum = pd.to_datetime(df["Month"].astype(str), format="%B", errors="coerce").dt.month
        except Exception:
            mnum = pd.Series([np.nan]*len(df))

        # if still NaN, try abbreviated names or numeric
        if mnum.isna().any():
            # try abbreviated names (Jan, Feb, …)
            try:
                mnum2 = pd.to_datetime(df["Month"].astype(str), format="%b", errors="coerce").dt.month
            except Exception:
                mnum2 = pd.Series([np.nan]*len(df))

            # combine
            mnum = mnum.fillna(mnum2)

        # if still NaN, try numeric conversion
        if mnum.isna().any():
            mnum3 = pd.to_numeric(df["Month"], errors="coerce")
            mnum = mnum.fillna(mnum3)

        df["Month_Num"] = mnum
        return df

    fail("No month column found. Please include either 'Month' (name or number) or 'Month_Num' in the CSV.")

def prepare_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a Year column exists. If missing, assume single-year and set to 2024.
    """
    df = df.copy()
    if "Year" not in df.columns:
        print("⚠️  'Year' column not found — assuming Year=2024 for all rows.")
        df["Year"] = 2024
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df

def validate_columns(df: pd.DataFrame):
    cols = set(df.columns)
    missing = REQUIRED_BASE - cols
    if missing:
        fail(f"Missing required columns {missing}. Found columns: {sorted(cols)}")

    for c in ["Year", "Month_Num", "Rainfall", "SAIDI", "SAIFI"]:
        if c not in df.columns:
            fail(f"Column '{c}' missing after preparation. Found: {sorted(df.columns)}")

def clean_types_and_dropna(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Month_Num"] = pd.to_numeric(df["Month_Num"], errors="coerce")
    df["Rainfall"]  = pd.to_numeric(df["Rainfall"], errors="coerce")
    df["SAIDI"]     = pd.to_numeric(df["SAIDI"], errors="coerce")
    df["SAIFI"]     = pd.to_numeric(df["SAIFI"], errors="coerce")
    df["Year"]      = pd.to_numeric(df["Year"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["Year","Month_Num","Rainfall","SAIDI","SAIFI"]).copy()
    after = len(df)
    if after < before:
        print(f"⚠️  Dropped {before - after} rows with missing required numeric values.")
    df["Year"] = df["Year"].astype(int)
    df["Month_Num"] = df["Month_Num"].astype(int)
    return df

def train_and_report(df: pd.DataFrame):
    """
    Train two LinearRegression models:
      - features = [Year, Month_Num, Rainfall]
      - targets  = SAIDI, SAIFI
    Print a tiny fit report (R², RMSE) on training data (for sanity).
    """
    X = df[["Year", "Month_Num", "Rainfall"]].to_numpy(dtype=float)

    models = {}
    report = {}

    for target in ["SAIDI", "SAIFI"]:
        y = df[target].to_numpy(dtype=float)
        model = LinearRegression().fit(X, y)
        yhat = model.predict(X)
        r2 = r2_score(y, yhat)
        rmse = float(np.sqrt(mean_squared_error(y, yhat)))

        models[target] = model
        report[target] = {"R2": r2, "RMSE": rmse,
                          "coef": model.coef_.tolist(),
                          "intercept": float(model.intercept_)}

    print("\n=== Training report (on provided data) ===")
    for target, stats in report.items():
        print(f"{target}: R²={stats['R2']:.3f}  RMSE={stats['RMSE']:.3f}  "
              f"coef={stats['coef']}  intercept={stats['intercept']:.3f}")

    return models

def main():
    print(f"📄 Loading: {TRAIN_CSV}")
    df = load_training_csv(TRAIN_CSV)

    # Build Month_Num and Year
    df = prepare_month(df)
    df = prepare_year(df)

    # Validate required columns after preparation
    validate_columns(df)

    # Clean numeric types and drop NAs
    df = clean_types_and_dropna(df)

    if len(df) < 12:
        print("⚠️  Warning: Very few rows detected. "
              "For reliable multi-year models you typically want several years × 12 months.")

    print("\nPreview of prepared data:")
    print(df.head())

    # Train models
    models = train_and_report(df)

    # Save pickle bundle
    out = Path(OUT_PKL)
    with open(out, "wb") as f:
        pickle.dump({
            "model_saidi": models["SAIDI"],
            "model_saifi": models["SAIFI"],
            "features": ["Year", "Month_Num", "Rainfall"]
        }, f)

    print(f"\n✅ Saved: {out.resolve()}")

if __name__ == "__main__":
    main()

