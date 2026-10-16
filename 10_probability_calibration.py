from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

for folder in [MODEL_DIR, RESULTS_DIR, PLOTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print("Loading final feature dataset...")

df = pd.read_parquet(DATA_DIR / "final_features.parquet")

# Count historical trips for each day/hour combination.
hourly_demand = (
    df.groupby(["day", "hour"])
    .size()
    .reset_index(name="trip_count")
)

print("Hourly demand records:", len(hourly_demand))

if len(hourly_demand) < 10:
    raise ValueError("Not enough hourly demand records to train the model.")

# Assign demand categories based on historical trip-count quantiles.
low_limit = hourly_demand["trip_count"].quantile(0.33)
high_limit = hourly_demand["trip_count"].quantile(0.66)

if low_limit == high_limit:
    raise ValueError(
        "Demand thresholds are identical. Check the trip-count distribution."
    )

hourly_demand["demand_class"] = 1

hourly_demand.loc[
    hourly_demand["trip_count"] <= low_limit,
    "demand_class"
] = 0

hourly_demand.loc[
    hourly_demand["trip_count"] > high_limit,
    "demand_class"
] = 2

print("\nDemand category counts:")
print(hourly_demand["demand_class"].value_counts().sort_index())

# Input features used by the demand model.
X = hourly_demand[["day", "hour"]]
y = hourly_demand["demand_class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

classifier.fit(X_train, y_train)

original_predictions = classifier.predict(X_test)

print(
    "\nOriginal model accuracy:",
    round(accuracy_score(y_test, original_predictions), 4)
)

# Calibrate probabilities using training data only.
calibrated_model = CalibratedClassifierCV(
    estimator=classifier,
    method="sigmoid",
    cv=3
)

calibrated_model.fit(X_train, y_train)

predictions = calibrated_model.predict(X_test)
probabilities = calibrated_model.predict_proba(X_test)

print(
    "Calibrated model accuracy:",
    round(accuracy_score(y_test, predictions), 4)
)

print("\nClassification report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=["Low Demand", "Medium Demand", "High Demand"],
        zero_division=0
    )
)

# Save classification metrics.
report = classification_report(
    y_test,
    predictions,
    labels=[0, 1, 2],
    target_names=["Low Demand", "Medium Demand", "High Demand"],
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    RESULTS_DIR / "demand_classification_report.csv"
)

# Draw one-vs-rest calibration curves.
plt.figure(figsize=(8, 6))

for class_number, class_name in enumerate(
    ["Low Demand", "Medium Demand", "High Demand"]
):
    actual = (y_test == class_number).astype(int)
    class_probability = probabilities[:, class_number]

    fraction_positive, mean_probability = calibration_curve(
        actual,
        class_probability,
        n_bins=5,
        strategy="quantile"
    )

    plt.plot(
        mean_probability,
        fraction_positive,
        marker="o",
        label=class_name
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Fraction")
plt.title("RideRush Demand Probability Calibration")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "calibration_curve.png",
    dpi=150
)

plt.close()

# Save the model artifact used by dashboard.py.
joblib.dump(
    calibrated_model,
    MODEL_DIR / "calibrated_demand_model.pkl"
)

print("\nDemand calibration completed.")
print("Saved models/calibrated_demand_model.pkl")
print("Saved results/demand_classification_report.csv")
print("Saved plots/calibration_curve.png")