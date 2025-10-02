import pandas as pd
import numpy as np
import pickle
from pathlib import Path

MONTH_MAP = {
    "jan":1, "january":1,
    "feb":2, "february":2,
    "mar":3, "march":3,
    "apr":4, "april":4,
    "may":5,
    "jun":6, "june":6,
    "jul":7, "july":7,
    "aug":8, "august":8,
    "sep":9, "september":9,
    "oct":10,"october":10,
    "nov":11,"november":11,
    "dec":12,"december":12,
    # allow numeric-like headers
    "1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"10":10,"11":11,"12":12
}

def _load_multiyear_models(pkl_path="saidi_saifi_multiyear.pkl"):
    p = Path(pkl_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Multi-year model not found: {p}. Run `python makepickle_multiyear.py` "
            "on your historical 2019–2024 dataset first."
        )
    with open(p, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model_saidi"], bundle["model_saifi"]

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _is_long_format(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    return {"year", "month", "rainfall"}.issubset(cols)

def _wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Convert wide (Year + Jan..Dec columns) to long (Year, Month_Num, Rainfall)."""
    df_wide = _normalize_cols(df_wide)
    # detect year column
    year_col = None
    for c in df_wide.columns:
        if c.lower() == "year":
            year_col = c
            break
    if year_col is None:
        raise ValueError("Could not find a 'Year' column in the forecast CSV.")

    # pick month columns by MONTH_MAP keys present
    month_cols = []
    for c in df_wide.columns:
        if c == year_col:
            continue
        key = str(c).strip().lower()
        if key in MONTH_MAP:
            month_cols.append(c)
    if not month_cols:
        raise ValueError(
            "No month columns found. Expect columns like Jan, February, 1..12, etc."
        )

    # build long rows
    rows = []
    for _, r in df_wide.iterrows():
        year_val = int(r[year_col])
        for mc in month_cols:
            key = str(mc).strip().lower()
            mnum = MONTH_MAP[key]
            try:
                rain = float(r[mc])
            except Exception:
                rain = np.nan
            rows.append({"Year": year_val, "Month_Num": mnum, "Rainfall": rain})
    out = pd.DataFrame(rows)
    out = out.dropna(subset=["Rainfall"])
    out = out.sort_values(["Year", "Month_Num"]).reset_index(drop=True)
    return out

def _long_to_long(df_long: pd.DataFrame) -> pd.DataFrame:
    """Standardize long format (Year, Month, Rainfall) to (Year, Month_Num, Rainfall)."""
    df_long = _normalize_cols(df_long)
    # map to canonical names
    rename_map = {}
    for c in df_long.columns:
        lc = c.lower()
        if lc == "year":
            rename_map[c] = "Year"
        elif lc == "month":
            rename_map[c] = "Month"
        elif lc in ("rain", "rainfall_mm", "rainfall"):
            rename_map[c] = "Rainfall"
    df_long = df_long.rename(columns=rename_map)

    if not {"Year", "Month", "Rainfall"}.issubset(df_long.columns):
        raise ValueError("Long format must contain columns: Year, Month, Rainfall")

    # Month can be name or number
    def month_to_num(val):
        s = str(val).strip().lower()
        if s in MONTH_MAP:
            return MONTH_MAP[s]
        try:
            n = int(float(s))
            if 1 <= n <= 12:
                return n
        except Exception:
            pass
        # try datetime
        try:
            return pd.to_datetime(s, format="%B").month
        except Exception:
            return np.nan

    df_long["Month_Num"] = df_long["Month"].map(month_to_num)
    out = df_long[["Year", "Month_Num", "Rainfall"]].dropna(subset=["Month_Num", "Rainfall"])
    out["Month_Num"] = out["Month_Num"].astype(int)
    out = out.sort_values(["Year", "Month_Num"]).reset_index(drop=True)
    return out

def parse_forecast_csv(path: str) -> pd.DataFrame:
    """Accepts:
       • Wide:  Year, Jan, Feb, ..., Dec
       • Long:  Year, Month, Rainfall
    Returns: DataFrame with columns [Year, Month_Num, Rainfall]
    """
    df = pd.read_csv(path)
    df = _normalize_cols(df)
    if _is_long_format(df):
        return _long_to_long(df)
    return _wide_to_long(df)

def batch_predict_from_csv(forecast_csv_path: str, model_path: str = "saidi_saifi_multiyear.pkl") -> pd.DataFrame:
    """Load forecast CSV (wide or long), run predictions with multi-year models."""
    model_saidi, model_saifi = _load_multiyear_models(model_path)
    df_future = parse_forecast_csv(forecast_csv_path)

    # ensure integer dtypes
    df_future["Year"] = df_future["Year"].astype(int)
    df_future["Month_Num"] = df_future["Month_Num"].astype(int)

    X = df_future[["Year", "Month_Num", "Rainfall"]]
    df_future["Pred_SAIDI"] = model_saidi.predict(X)
    df_future["Pred_SAIFI"] = model_saifi.predict(X)
    return df_future[["Year", "Month_Num", "Rainfall", "Pred_SAIDI", "Pred_SAIFI"]]

def auto_future_from_last_year(history_csv_path: str, model_path: str = "saidi_saifi_multiyear.pkl") -> pd.DataFrame:
    """Fallback builder: use the last available year in history as 'baseline' rainfall for next year."""
    hist = pd.read_csv(history_csv_path)
    # Expect Month column OR Month_Num; try to normalize
    if "Month_Num" not in hist.columns:
        try:
            hist["Month_Num"] = pd.to_datetime(hist["Month"], format="%B").dt.month
        except Exception:
            hist["Month_Num"] = pd.to_numeric(hist.get("Month"), errors="coerce")
    hist = hist.dropna(subset=["Month_Num", "Rainfall"])

    last_year = int(hist["Year"].max())
    base = hist[hist["Year"] == last_year][["Month_Num", "Rainfall"]].copy()
    base["Year"] = last_year + 1
    base = base[["Year", "Month_Num", "Rainfall"]].sort_values("Month_Num")

    model_saidi, model_saifi = _load_multiyear_models(model_path)
    X = base[["Year", "Month_Num", "Rainfall"]]
    base["Pred_SAIDI"] = model_saidi.predict(X)
    base["Pred_SAIFI"] = model_saifi.predict(X)
    return base



