import pickle
import numpy as np

def predict(month_num, rainfall_mm):
    """Predict SAIDI & SAIFI for a given month and rainfall value."""
    with open("saidi_saifi_model.pkl", "rb") as f:
        models = pickle.load(f)

    X_new = np.array([[month_num, rainfall_mm]])
    results = {}
    for metric, model in models.items():
        results[metric] = float(model.predict(X_new)[0])
    return results

if __name__ == "__main__":
    # Example usage: Predict for July (7) with 250 mm rainfall
    month_num = 7
    rainfall = 250.0

    output = predict(month_num, rainfall)
    print(f"📅 Month {month_num}, 🌧 Rainfall {rainfall} mm")
    print(f"Predicted SAIDI = {output['SAIDI']:.2f}")
    print(f"Predicted SAIFI = {output['SAIFI']:.2f}")
