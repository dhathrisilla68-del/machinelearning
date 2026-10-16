import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

print("Loading final feature dataset...")

df = pd.read_parquet("data/final_features.parquet")

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

models = {
    "Ridge": Ridge(),
    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    print(name, "training completed!")

    file_name = name.lower().replace(" ", "_")

    joblib.dump(
        model,
        f"models/{file_name}_model.pkl"
    )

print("\nAll models trained successfully!")
print("Models saved in models/ folder.")