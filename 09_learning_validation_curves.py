from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve, validation_curve


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "trip_distance",
    "passenger_count",
    "hour",
    "day",
    "month",
    "day_of_week",
    "is_weekend",
    "is_rush_hour"
]

print("Loading feature dataset...")

df = pd.read_parquet(DATA_DIR / "final_features.parquet")

if len(df) > 20000:
    df = df.sample(n=20000, random_state=42)

X = df[FEATURES]
y = df["trip_duration"]

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# -----------------------------------------------------
# LEARNING CURVE
# -----------------------------------------------------

print("\nCreating learning curve...")

train_sizes, train_scores, validation_scores = learning_curve(
    model,
    X,
    y,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

train_mae = -train_scores.mean(axis=1)
validation_mae = -validation_scores.mean(axis=1)

learning_results = pd.DataFrame({
    "Training Samples": train_sizes,
    "Training MAE": train_mae,
    "Validation MAE": validation_mae
})

print(learning_results.to_string(index=False))

learning_results.to_csv(
    BASE_DIR / "results" / "learning_curve_results.csv",
    index=False
)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mae, marker="o", label="Training MAE")
plt.plot(train_sizes, validation_mae, marker="o", label="Validation MAE")
plt.xlabel("Training Samples")
plt.ylabel("MAE (minutes)")
plt.title("RideRush Learning Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "learning_curve.png", dpi=150)
plt.close()

# -----------------------------------------------------
# VALIDATION CURVE
# -----------------------------------------------------

print("\nCreating validation curve...")

depth_values = [5, 8, 10, 12, 15, 20]

train_scores, validation_scores = validation_curve(
    model,
    X,
    y,
    param_name="max_depth",
    param_range=depth_values,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

train_mae = -train_scores.mean(axis=1)
validation_mae = -validation_scores.mean(axis=1)

validation_results = pd.DataFrame({
    "Max Depth": depth_values,
    "Training MAE": train_mae,
    "Validation MAE": validation_mae
})

print(validation_results.to_string(index=False))

validation_results.to_csv(
    BASE_DIR / "results" / "validation_curve_results.csv",
    index=False
)

plt.figure(figsize=(8, 5))
plt.plot(depth_values, train_mae, marker="o", label="Training MAE")
plt.plot(depth_values, validation_mae, marker="o", label="Validation MAE")
plt.xlabel("Maximum Tree Depth")
plt.ylabel("MAE (minutes)")
plt.title("RideRush Validation Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "validation_curve.png", dpi=150)
plt.close()

print("\nLearning and validation curve analysis completed.")
print("Saved plots/learning_curve.png")
print("Saved plots/validation_curve.png")