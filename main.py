import pandas as pd
from functions import validate_data, run_correlations, run_regression

def main():
    path = "Average national rainfall and SAIDI-SAIFI_2024.csv"
    df = pd.read_csv(path)
    df = validate_data(df)

    corr_results = []
    reg_results = []

    # --- Overall correlations/regressions ---
    for metric in ["SAIDI", "SAIFI"]:
        corr_results.append(run_correlations(df["Rainfall"], df[metric], metric, "Overall"))
        reg_results.append(run_regression(df["Rainfall"], df[metric], metric, "Overall", make_plots=True))

    # --- Per-month correlations/regressions ---
    for m in df["Month_Num"].unique():
        df_m = df[df["Month_Num"] == m]
        month_name = df_m["Month"].iloc[0]
        for metric in ["SAIDI", "SAIFI"]:
            corr_results.append(run_correlations(df_m["Rainfall"], df_m[metric], metric, f"Month {month_name}"))
            reg_results.append(run_regression(df_m["Rainfall"], df_m[metric], metric, f"Month {month_name}", make_plots=False))

    # Save results
    pd.DataFrame(corr_results).to_csv("correlation_results.csv", index=False)
    pd.DataFrame(reg_results).to_csv("regression_results.csv", index=False)
    print("\n✅ Results saved: correlation_results.csv and regression_results.csv")

if __name__ == "__main__":
    main()
