from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

# Use a reproducible evaluation sample.
if len(df) > 500000:
    df = df.sample(n=500000, random_state=42)

X = df[FEATURES]
y = df["trip_duration"]

# Keep one common test set for every model.
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

X_train, X_validation, y_train, y_validation = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

print("Training:", len(X_train))
print("Validation:", len(X_validation))
print("Testing:", len(X_test))

model_files = {
    "Ridge Regression": "ridge_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl"
}

results = []
loaded_models = {}

for name, filename in model_files.items():
    path = MODEL_DIR / filename

    if not path.exists():
        print(f"Skipping {name}: missing file {path}")
        continue

    model = joblib.load(path)
    loaded_models[name] = model

    predictions = model.predict(X_validation)

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_validation, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_validation, predictions)),
        "R2": r2_score(y_validation, predictions)
    })

if not results:
    raise FileNotFoundError(
        "No trained models found. Run your model training script first."
    )

results_df = pd.DataFrame(results).sort_values("MAE")

print("\nValidation model comparison:")
print(results_df.to_string(index=False))

results_df.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)

# Five-fold cross-validation for Random Forest, if available.
if "Random Forest" in loaded_models:
    print("\nRandom Forest 5-fold cross-validation:")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        loaded_models["Random Forest"],
        X_train,
        y_train,
        cv=kf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    fold_mae = -scores

    for i, score in enumerate(fold_mae, start=1):
        print(f"Fold {i} MAE: {score:.4f}")

    print(f"Average CV MAE: {fold_mae.mean():.4f}")
    print(f"CV MAE standard deviation: {fold_mae.std():.4f}")

# Final held-out test result for Random Forest.
if "Random Forest" in loaded_models:
    model = loaded_models["Random Forest"]
    predictions = model.predict(X_test)

    test_results = pd.DataFrame([{
        "Model": "Random Forest",
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions)
    }])

    test_results.to_csv(
        RESULTS_DIR / "random_forest_test_results.csv",
        index=False
    )

    print("\nFinal test results:")
    print(test_results.to_string(index=False))

print("\nEvaluation completed.")
print("Saved: results/model_comparison.csv")