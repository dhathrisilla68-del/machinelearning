import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("Loading final feature dataset...")

df = pd.read_parquet(
    "data/final_features.parquet"
)

print("Original dataset shape:", df.shape)


# ==================================================
# USE SAMPLE FOR FASTER TRAINING
# ==================================================

sample_size = 50000

if len(df) > sample_size:
    df = df.sample(
        n=sample_size,
        random_state=42
    )

print("Tuning dataset shape:", df.shape)


# ==================================================
# FEATURES AND TARGET
# ==================================================

features = [
    "trip_distance",
    "passenger_count",
    "hour",
    "day",
    "month",
    "day_of_week",
    "is_weekend",
    "is_rush_hour"
]

X = df[features]

y = df["trip_duration"]


# ==================================================
# TRAIN AND TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==================================================
# RANDOM FOREST MODEL
# ==================================================

print("\n==============================")
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("==============================")


random_forest = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)


# ==================================================
# PARAMETER SEARCH
# ==================================================

parameter_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}


print("\nStarting Randomized Search...")

search = RandomizedSearchCV(
    estimator=random_forest,
    param_distributions=parameter_grid,
    n_iter=8,
    cv=3,
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=-1,
    verbose=1
)


search.fit(
    X_train,
    y_train
)


# ==================================================
# BEST PARAMETERS
# ==================================================

print("\n==============================")
print("BEST PARAMETERS")
print("==============================")

print(
    search.best_params_
)

print(
    "\nBest Cross-Validation MAE:",
    round(
        -search.best_score_,
        4
    )
)


# ==================================================
# BEST MODEL
# ==================================================

best_model = search.best_estimator_


print("\nBest Random Forest model selected.")


# ==================================================
# TEST EVALUATION
# ==================================================

print("\n==============================")
print("TEST SET EVALUATION")
print("==============================")


y_pred = best_model.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


# MAPE
y_test_array = np.array(y_test)

non_zero = y_test_array != 0

mape = np.mean(
    np.abs(
        (
            y_test_array[non_zero]
            - y_pred[non_zero]
        )
        / y_test_array[non_zero]
    )
) * 100


print(
    "MAE :",
    round(mae, 4)
)

print(
    "RMSE:",
    round(rmse, 4)
)

print(
    "MAPE:",
    round(mape, 2),
    "%"
)

print(
    "R2  :",
    round(r2, 4)
)


# ==================================================
# SAVE TUNED MODEL
# ==================================================

joblib.dump(
    best_model,
    "models/tuned_random_forest_model.pkl"
)

print(
    "\nTuned model saved successfully!"
)

print(
    "File: models/tuned_random_forest_model.pkl"
)


# ==================================================
# SAVE RESULTS
# ==================================================

results = pd.DataFrame({
    "Model": [
        "Tuned Random Forest"
    ],
    "MAE": [
        mae
    ],
    "RMSE": [
        rmse
    ],
    "MAPE": [
        mape
    ],
    "R2": [
        r2
    ]
})


results.to_csv(
    "results/tuned_model_results.csv",
    index=False
)


print(
    "\nTuning results saved!"
)

print(
    "File: results/tuned_model_results.csv"
)


# ==================================================
# FEATURE IMPORTANCE
# ==================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": best_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

print(
    importance
)


importance.to_csv(
    "results/feature_importance.csv",
    index=False
)


print(
    "\nFeature importance saved!"
)


# ==================================================
# FINAL MESSAGE
# ==================================================

print("\n==============================")
print("M5 HYPERPARAMETER TUNING COMPLETED")
print("==============================")