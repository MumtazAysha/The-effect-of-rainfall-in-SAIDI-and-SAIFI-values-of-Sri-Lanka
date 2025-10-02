import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr 
from sklearn.metrics import r2_score,mean_squared_error
from future_predict import batch_predict_from_csv


from future_predict import batch_predict_from_csv, auto_future_from_last_year

def correlation_summary(df: pd.DataFrame, y_col: str):
    """
    Returns a dict with Pearson r, Spearman rho, R2, RMSE for Rainfall -> y_col.
    Expects df to have numeric columns: Rainfall, y_col.
    """
    # guard
    d = df.dropna(subset=["Rainfall", y_col]).copy()
    if len(d) < 3:
        return {"n": len(d), "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_rho": np.nan, "spearman_p": np.nan,
                "r2": np.nan, "rmse": np.nan}

    x = d["Rainfall"].to_numpy(dtype=float).reshape(-1, 1)
    y = d[y_col].to_numpy(dtype=float)

    # correlations
    pr, pp = pearsonr(d["Rainfall"].to_numpy(dtype=float), y)
    sr, sp = spearmanr(d["Rainfall"].to_numpy(dtype=float), y)

    # simple linear regression for R2 / RMSE
    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

    return {"n": len(d), "pearson_r": pr, "pearson_p": pp,
            "spearman_rho": sr, "spearman_p": sp,
            "r2": r2, "rmse": rmse}

# ---------------------------
# Config
# ---------------------------
MODEL_PATH = "saidi_saifi_model.pkl"
DATA_PATH  = "Average national rainfall and SAIDI-SAIFI_2024.csv"

MONTH_MAP = {
    1:"January",2:"February",3:"March",4:"April",
    5:"May",6:"June",7:"July",8:"August",
    9:"September",10:"October",11:"November",12:"December"
}

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Rainfall → SAIDI/SAIFI Dashboard", page_icon="🌧️", layout="wide")

# Custom CSS
st.markdown("""
<style>
h1, h2, h3 { color: #0d47a1; }
.stButton > button {
    border-radius: 8px;
    background-color: #1e88e5;
    color: white;
    font-size: 16px;
}
.metric-card {
    background-color:#f8f9fa;
    padding:20px;
    border-radius:12px;
    text-align:center;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
    margin: 5px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🌧️ Rainfall → SAIDI/SAIFI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Analyze & Predict Power Outages from Rainfall Patterns</p>", unsafe_allow_html=True)

# ---------------------------
# Load models bundle
# ---------------------------
@st.cache_resource
def load_models_bundle(pkl_path: str):
    if not Path(pkl_path).exists():
        st.error(f"Model file not found: {pkl_path}. Please run `python makepickle.py` first.")
        st.stop()
    import pickle
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle

bundle = load_models_bundle(MODEL_PATH)
models_full = bundle["full"]
models_month_only = bundle["month_only"]
baseline_rain = bundle["baseline_rainfall_by_month"]

# ---------------------------
# Load 2024 data
# ---------------------------
@st.cache_data
def load_2024(path: str):
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    try:
        df["Month_Num"] = pd.to_datetime(df["Month"], format="%B").dt.month
    except Exception:
        df["Month_Num"] = pd.to_numeric(df["Month"], errors="coerce")
    df = df.dropna(subset=["Month_Num", "Rainfall", "SAIDI", "SAIFI"]).copy()
    df["Month_Num"] = df["Month_Num"].astype(int)
    return df

df2024 = load_2024(DATA_PATH)

def correlation_summary(df: pd.DataFrame, y_col: str):
    """
    Returns Pearson r, Spearman rho, R2, RMSE for Rainfall -> y_col.
    """
    d = df.dropna(subset=["Rainfall", y_col]).copy()
    if len(d) < 3:
        return {
            "n": len(d),
            "pearson_r": np.nan, "pearson_p": np.nan,
            "spearman_rho": np.nan, "spearman_p": np.nan,
            "r2": np.nan, "rmse": np.nan
        }

    x = d["Rainfall"].to_numpy(dtype=float).reshape(-1, 1)
    y = d[y_col].to_numpy(dtype=float)

    pr, pp = pearsonr(d["Rainfall"].to_numpy(dtype=float), y)
    sr, sp = spearmanr(d["Rainfall"].to_numpy(dtype=float), y)

    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

    return {
        "n": len(d),
        "pearson_r": pr, "pearson_p": pp,
        "spearman_rho": sr, "spearman_p": sp,
        "r2": r2, "rmse": rmse
    }


# Tabs

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔮 Single Prediction",
    "📊 Overall Regression",
    "📈 Per-Month Scatter",
    "🚀 Future Predictions",
    "🔍 Correlation Summary",
    "🌍 Multi-Year Predictions"
])

def predict_with_multiyear(df_future: pd.DataFrame, model_path="saidi_saifi_multiyear.pkl") -> pd.DataFrame:
    """Predict SAIDI/SAIFI given df_future[Year, Month_Num, Rainfall] using multi-year pickle."""
    p = Path(model_path)
    if not p.exists():
        st.error(f"Multi-year model not found: {p}. Run your trainer to create it.")
        st.stop()
    with open(p, "rb") as f:
        bundle = pickle.load(f)
    m_saidi = bundle["model_saidi"]
    m_saifi = bundle["model_saifi"]
    X = df_future[["Year", "Month_Num", "Rainfall"]].to_numpy(dtype=float)
    df_future = df_future.copy()
    df_future["Pred_SAIDI"] = m_saidi.predict(X)
    df_future["Pred_SAIFI"] = m_saifi.predict(X)
    return df_future

def month_labels():
    return ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
# Tab 1: Single Prediction

with tab1:
    st.header("Single Prediction")

    col1, col2 = st.columns(2)
    with col1:
        month_num = st.selectbox(
            "Select Month",
            options=list(MONTH_MAP.keys()),
            format_func=lambda m: MONTH_MAP[m],
            index=6
        )

    mode = st.radio(
        "Rainfall input mode",
        ["I know the rainfall", "I don't know the rainfall"],
        index=0,
        horizontal=True
    )

    if mode == "I know the rainfall":
        rainfall_input = st.text_input("Enter Rainfall (mm)", value="0")
        try:
            rainfall = float(rainfall_input)
            X_full = np.array([[month_num, rainfall]])
            saidi_pred = float(models_full["SAIDI"].predict(X_full)[0])
            saifi_pred = float(models_full["SAIFI"].predict(X_full)[0])

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"<div class='metric-card'><h4>Predicted SAIDI</h4><h2>{saidi_pred:.2f}</h2></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='metric-card'><h4>Predicted SAIFI</h4><h2>{saifi_pred:.3f}</h2></div>", unsafe_allow_html=True)

        except ValueError:
            st.error("⚠️ Please enter a valid numeric rainfall value (e.g., 123.45)")

    else:
        imputed_rain = float(baseline_rain.get(month_num, 0.0))
        X_full = np.array([[month_num, imputed_rain]])
        saidi_full = float(models_full["SAIDI"].predict(X_full)[0])
        saifi_full = float(models_full["SAIFI"].predict(X_full)[0])

        X_mo = np.array([[month_num]])
        saidi_mo = float(models_month_only["SAIDI"].predict(X_mo)[0])
        saifi_mo = float(models_month_only["SAIFI"].predict(X_mo)[0])

        st.info(f"No rainfall entered. Using baseline rainfall for {MONTH_MAP[month_num]}: {imputed_rain:.1f} mm")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<div class='metric-card'><h4>SAIDI (imputed rain)</h4><h2>{saidi_full:.2f}</h2></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><h4>SAIDI (month-only)</h4><h2>{saidi_mo:.2f}</h2></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><h4>SAIFI (imputed rain)</h4><h2>{saifi_full:.3f}</h2></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><h4>SAIFI (month-only)</h4><h2>{saifi_mo:.3f}</h2></div>", unsafe_allow_html=True)

# Tab 2: Overall Regression

with tab2:
    st.header("Overall Regression (2024)")
    if df2024 is not None and len(df2024) > 2:
        for metric in ["SAIDI", "SAIFI"]:
            x = df2024["Rainfall"].to_numpy(dtype=float).reshape(-1, 1)
            y = df2024[metric].to_numpy(dtype=float)
            model = LinearRegression().fit(x, y)
            y_pred = model.predict(x)
            r2 = r2_score(y, y_pred)

            order = np.argsort(x.flatten())
            x_sorted, y_pred_sorted = x.flatten()[order], y_pred[order]

            fig, ax = plt.subplots(figsize=(6,4))  # smaller figure
            ax.scatter(df2024["Rainfall"], y, label="Actual")
            ax.plot(x_sorted, y_pred_sorted, label=f"Regression (R²={r2:.2f})", color="red", linewidth=2)
            ax.set_xlabel("Rainfall (mm)")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} vs Rainfall — 2024")
            ax.legend()
            st.pyplot(fig, use_container_width=False)  # do not stretch
    else:
        st.warning("Not enough data to show regression plots.")

# =========================================================
# Tab 3: Per-Month Scatter
# =========================================================
with tab3:
    st.header("Per-Month Scatter (2024)")
    if df2024 is not None and len(df2024) > 0:
        month_choice = st.selectbox("Choose Month", options=list(MONTH_MAP.keys()), format_func=lambda m: MONTH_MAP[m])
        mdf = df2024[df2024["Month_Num"] == month_choice]
        if len(mdf) > 0:
            st.subheader(f"{MONTH_MAP[month_choice]}")
            for metric in ["SAIDI", "SAIFI"]:
                # Build a small grid for fitted line using the trained full models
                x_line = np.linspace(max(0.0, mdf["Rainfall"].min()*0.8), max(10.0, mdf["Rainfall"].max()*1.2), 50)
                X_line = np.column_stack([np.full_like(x_line, month_choice), x_line])
                y_line = models_full[metric].predict(X_line)

                fig, ax = plt.subplots(figsize=(6,4))  # smaller figure
                ax.scatter(mdf["Rainfall"], mdf[metric], label="Actual")
                ax.plot(x_line, y_line, label="Model fit", color="red")
                ax.set_xlabel("Rainfall (mm)")
                ax.set_ylabel(metric)
                ax.set_title(f"{metric} vs Rainfall — {MONTH_MAP[month_choice]} (2024)")
                ax.legend()
                st.pyplot(fig, use_container_width=False)  # do not stretch
        else:
            st.info(f"No data available for {MONTH_MAP[month_choice]}")
    else:
        st.warning("Upload the 2024 CSV file to see per-month scatter plots.")

with tab4:
    st.header("Future Predictions")
    tabA, tabB = st.tabs(["From Forecast CSV (Option A)", "Auto Baseline (Option B)"])

    with tabA:
        forecast_file = st.file_uploader("Upload forecast CSV (Month_Num,Rainfall)", type=["csv"])
        if forecast_file is not None:
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(forecast_file.read())
                temp_path = tmp.name
            try:
                out_df = batch_predict_from_csv(temp_path, model_path=MODEL_PATH)
                st.success("✅ Predictions generated")
                st.dataframe(out_df)
                st.download_button("⬇️ Download predictions", data=out_df.to_csv(index=False), file_name="forecast_predictions.csv")
                # Optional: small line plots of predicted SAIDI/SAIFI
                for metric in ["Pred_SAIDI", "Pred_SAIFI"]:
                    fig, ax = plt.subplots(figsize=(6,4))
                    ax.plot(out_df["Month_Num"], out_df[metric], marker="o")
                    ax.set_xticks(range(1,13))
                    ax.set_xlabel("Month")
                    ax.set_ylabel(metric.replace("Pred_",""))
                    ax.set_title(f"{metric.replace('Pred_','')} (Forecast)")
                    st.pyplot(fig, use_container_width=False)
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.remove(temp_path)
    

    with tabB:
        if st.button("Generate baseline predictions"):
            try:
                out_df = auto_future_from_last_year(DATA_PATH, MODEL_PATH)
                st.success("✅ Baseline predictions generated")
                st.dataframe(out_df)
                st.download_button("⬇️ Download baseline", data=out_df.to_csv(index=False), file_name="baseline_predictions.csv")
                # Optional: small line plots of predicted SAIDI/SAIFI
                for metric in ["Pred_SAIDI", "Pred_SAIFI"]:
                    fig, ax = plt.subplots(figsize=(6,4))
                    ax.plot(out_df["Month_Num"], out_df[metric], marker="o")
                    ax.set_xticks(range(1,13))
                    ax.set_xlabel("Month")
                    ax.set_ylabel(metric.replace("Pred_",""))
                    ax.set_title(f"{metric.replace('Pred_','')} (Baseline)")
                    st.pyplot(fig, use_container_width=False)
            except Exception as e:
                st.error(f"Error: {e}")


with tab5:
    st.header("Correlation Summary")

    if df2024 is None or len(df2024) < 3:
        st.warning("Not enough data to compute correlations.")
    else:
 
        # Whole year correlation
    
        st.subheader("📊 Whole Year Correlation (All 12 months in 2024)")
        for metric in ["SAIDI", "SAIFI"]:
            res = correlation_summary(df2024, metric)

            st.markdown(f"**{metric}**")
            st.write(f"**n**: {res['n']}")
            st.write(f"**Pearson r**: {res['pearson_r']:.3f}, p = {res['pearson_p']:.3f}")
            st.write(f"**Spearman ρ**: {res['spearman_rho']:.3f}, p = {res['spearman_p']:.3f}")
            st.write(f"**R²**: {res['r2']:.3f}")
            st.write(f"**RMSE**: {res['rmse']:.3f}")

            if np.isfinite(res['pearson_r']) and abs(res['pearson_r']) >= 0.5 and res['pearson_p'] < 0.05:
                st.success(f"✅ Statistically significant correlation between Rainfall and {metric}.")
            else:
                st.info(f"ℹ️ No strong/clear correlation between Rainfall and {metric}.")

        
        # Monthly values inspector
   
        st.subheader("📅 Inspect Monthly Values")
        month_choice = st.selectbox(
            "Select Month",
            options=sorted(df2024["Month_Num"].unique().tolist()),
            format_func=lambda m: MONTH_MAP.get(m, str(m))
        )

        dff = df2024[df2024["Month_Num"] == month_choice]
        if not dff.empty:
            st.write(f"**Month:** {MONTH_MAP[month_choice]}")
            st.write(f"Rainfall = {float(dff['Rainfall'].values[0]):.2f} mm")
            st.write(f"SAIDI = {float(dff['SAIDI'].values[0]):.2f}")
            st.write(f"SAIFI = {float(dff['SAIFI'].values[0]):.2f}")
# Month-by-Month Correlation (select a month)

        st.subheader("📆 Month-by-Month Correlation (across years)")

# Choose the source df: if you later load multi-year data into df_all, this will use it.
        df_for_corr = globals().get("df_all", df2024)

# Dropdown to pick the month
        month_pick = st.selectbox(
        "Select a month to compute correlation",
        options=sorted(df_for_corr["Month_Num"].dropna().astype(int).unique().tolist()),
        format_func=lambda m: MONTH_MAP.get(m, str(m))
)

# Filter to the chosen month across all years/rows available
        d_month = df_for_corr[df_for_corr["Month_Num"] == month_pick].dropna(subset=["Rainfall", "SAIDI", "SAIFI"])

# Need at least ~3 rows to compute meaningful Pearson/Spearman/R²/RMSE
        if len(d_month) < 3:
         st.warning(
         f"Not enough rows for **{MONTH_MAP.get(month_pick, month_pick)}** to compute correlation. "
         "Add more years of data (e.g., 2019–2024) so this becomes a multi-point analysis."
    )
        else:
         for metric in ["SAIDI", "SAIFI"]:
          res = correlation_summary(d_month, metric)

         st.markdown(f"**{MONTH_MAP.get(month_pick, month_pick)} — {metric}**")
         c1, c2, c3, c4, c5 = st.columns(5)
         c1.metric("n", f"{res['n']}")
         c2.metric("Pearson r", f"{res['pearson_r']:.3f}")
         c3.metric("Spearman ρ", f"{res['spearman_rho']:.3f}")
         c4.metric("R²", f"{res['r2']:.3f}")
         c5.metric("RMSE", f"{res['rmse']:.3f}")

        # Decide message based on Pearson result
        has_corr = (
            np.isfinite(res["pearson_r"]) and abs(res["pearson_r"]) >= 0.5 and
            np.isfinite(res["pearson_p"]) and res["pearson_p"] < 0.05
        )
        if has_corr:
            st.success(
                f"✅ Statistically significant correlation between Rainfall and **{metric}** "
                f"for **{MONTH_MAP.get(month_pick, month_pick)}** "
                f"(r = {res['pearson_r']:.3f}, p = {res['pearson_p']:.3f})."
            )
        else:
            st.info(
                f"ℹ️ No strong/clear linear correlation between Rainfall and **{metric}** "
                f"for **{MONTH_MAP.get(month_pick, month_pick)}** "
                f"(r = {res['pearson_r']:.3f}, p = {res['pearson_p']:.3f})."
            )

with tab6:
  
    st.header("🌍 Multi-Year Future Predictions (No CSV needed)")

    input_mode = st.radio(
        "Choose input method",
        ["Upload CSV", "Manual rainfall", "Auto baseline"],
        horizontal=True
    )

    # Common year picker
    year_choice = st.number_input("Prediction year", min_value=2025, max_value=2100, value=2025, step=1)

    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload Rainfall Forecast CSV (wide: Year, Jan..Dec  OR long: Year, Month, Rainfall)",
                                    type=["csv"])
        if uploaded is not None:
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded.read())
                temp_path = tmp.name
            try:
                out_df = batch_predict_from_csv(temp_path, model_path="saidi_saifi_multiyear.pkl")
                st.success("Predictions generated from CSV.")
                st.dataframe(out_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error while predicting from CSV: {e}")
            finally:
                try: os.remove(temp_path)
                except Exception: pass

    elif input_mode == "Manual rainfall":
        st.caption("Enter 12 rainfall values (mm) for each month.")
        # Text area input (simplest for users)
        vals = st.text_area(
            "Rainfall values (12 numbers, comma or space separated)",
            value=",".join([""]*12),
            help="Example: 200,180,170,190,210,240,260,250,230,220,210,200"
        )
        if st.button("Predict (Manual input)"):
            try:
                arr = [float(x) for x in vals.replace(",", " ").split()]
                if len(arr) != 12:
                    st.error("Please enter exactly 12 numbers.")
                else:
                    df_future = pd.DataFrame({
                        "Year": [year_choice]*12,
                        "Month_Num": list(range(1,13)),
                        "Rainfall": arr
                    })
                    out_df = predict_with_multiyear(df_future, model_path="saidi_saifi_multiyear.pkl")
                    st.success("Predictions generated.")
                    st.dataframe(out_df, use_container_width=True)

                    # Small plots
                    fig, ax = plt.subplots(figsize=(7,3.5))
                    ax.plot(out_df["Month_Num"], out_df["Pred_SAIDI"], marker="o", label="SAIDI")
                    ax.plot(out_df["Month_Num"], out_df["Pred_SAIFI"], marker="o", label="SAIFI")
                    ax.set_xticks(range(1,13)); ax.set_xticklabels(month_labels(), rotation=0)
                    ax.set_title(f"Predicted SAIDI & SAIFI — {year_choice}")
                    ax.set_ylabel("Predicted value"); ax.legend()
                    st.pyplot(fig, use_container_width=False)

                    st.download_button(
                        "⬇️ Download predictions (CSV)",
                        data=out_df.to_csv(index=False),
                        file_name=f"predictions_{year_choice}.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Could not parse manual input: {e}")

    else:  # Auto baseline
        st.caption("Build rainfall automatically from your training data.")
        auto_mode = st.selectbox(
            "Baseline source",
            ["Use last available year in training data", "Use monthly average across training years"]
        )

        # Ask for the training CSV so we can compute baselines (or let user point to it)
        hist_csv = st.text_input(
            "Path to your historical training CSV (used to compute rainfall baselines)",
            value="Rainfall-SAIDI-SAIFI.csv",
            help="This is the same file you used to build the multi-year pickle."
        )

        if st.button("Predict (Auto baseline)"):
            try:
                hist = pd.read_csv(hist_csv)
                # normalize columns
                cols = [c.strip() for c in hist.columns]
                hist.columns = cols

                # Ensure Month_Num present (accept Month or Month_Num)
                if "Month_Num" not in hist.columns:
                    if "Month" in hist.columns:
                        # Try to convert Month to number (name, abbrev, or numeric)
                        mnum = pd.to_datetime(hist["Month"].astype(str), format="%B", errors="coerce").dt.month
                        if mnum.isna().any():
                            mnum2 = pd.to_datetime(hist["Month"].astype(str), format="%b", errors="coerce").dt.month
                            mnum = mnum.fillna(mnum2)
                        if mnum.isna().any():
                            mnum3 = pd.to_numeric(hist["Month"], errors="coerce")
                            mnum = mnum.fillna(mnum3)
                        hist["Month_Num"] = mnum
                    else:
                        st.error("Training CSV needs either Month or Month_Num to build baselines.")
                        st.stop()

                hist = hist.dropna(subset=["Month_Num", "Rainfall"]).copy()
                hist["Month_Num"] = hist["Month_Num"].astype(int)

                if auto_mode == "Use last available year in training data":
                    if "Year" not in hist.columns:
                        st.error("Training CSV needs a Year column for this mode.")
                        st.stop()
                    last_year = int(hist["Year"].max())
                    base = hist[hist["Year"] == last_year].groupby("Month_Num", as_index=False)["Rainfall"].mean()
                else:
                    base = hist.groupby("Month_Num", as_index=False)["Rainfall"].mean()

                # Ensure we have all 12 months; if not, fill missing with overall mean
                all_months = pd.DataFrame({"Month_Num": list(range(1,13))})
                base = all_months.merge(base, on="Month_Num", how="left")
                if base["Rainfall"].isna().any():
                    base["Rainfall"] = base["Rainfall"].fillna(base["Rainfall"].mean())

                df_future = base.copy()
                df_future["Year"] = year_choice
                df_future = df_future[["Year","Month_Num","Rainfall"]]

                out_df = predict_with_multiyear(df_future, model_path="saidi_saifi_multiyear.pkl")
                st.success("Predictions generated from auto baseline.")
                st.dataframe(out_df, use_container_width=True)

                # Plot
                fig, ax = plt.subplots(figsize=(7,3.5))
                ax.plot(out_df["Month_Num"], out_df["Pred_SAIDI"], marker="o", label="SAIDI")
                ax.plot(out_df["Month_Num"], out_df["Pred_SAIFI"], marker="o", label="SAIFI")
                ax.set_xticks(range(1,13)); ax.set_xticklabels(month_labels(), rotation=0)
                ax.set_title(f"Predicted SAIDI & SAIFI — {year_choice} (Auto baseline)")
                ax.set_ylabel("Predicted value"); ax.legend()
                st.pyplot(fig, use_container_width=False)

                st.download_button(
                    "⬇️ Download predictions (CSV)",
                    data=out_df.to_csv(index=False),
                    file_name=f"predictions_{year_choice}_autobaseline.csv",
                    mime="text/csv"
                )
            except FileNotFoundError:
                st.error(f"Cannot find training CSV: {hist_csv}")
            except Exception as e:
                st.error(f"Auto-baseline failed: {e}")