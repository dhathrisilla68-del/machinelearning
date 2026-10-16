import pandas as pd

# Load cleaned dataset
input_file = "data/cleaned_taxi_data.parquet"

print("Loading cleaned dataset...")

df = pd.read_parquet(input_file)

print("Cleaned dataset shape:", df.shape)

# Convert pickup time
df["tpep_pickup_datetime"] = pd.to_datetime(
    df["tpep_pickup_datetime"]
)

# Create time-based features
df["hour"] = (
    df["tpep_pickup_datetime"].dt.hour
)

df["day"] = (
    df["tpep_pickup_datetime"].dt.day
)

df["month"] = (
    df["tpep_pickup_datetime"].dt.month
)

df["day_of_week"] = (
    df["tpep_pickup_datetime"].dt.dayofweek
)

# Weekend feature
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# Rush hour feature
df["is_rush_hour"] = (
    df["hour"].isin(
        [7, 8, 9, 16, 17, 18, 19]
    )
).astype(int)

# ML input features
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

# Target
target = "trip_duration"

# Create final dataset
final_df = df[
    features + [target]
]

print("\nFeature engineering completed!")

print("\nFinal dataset shape:")
print(final_df.shape)

print("\nML Features:")
print(features)

print("\nTarget:")
print(target)

print("\nFirst 5 rows:")
print(final_df.head())

print("\nFeature information:")
final_df.info()

# Save final feature dataset
output_file = (
    "data/final_features.parquet"
)

final_df.to_parquet(
    output_file,
    index=False
)

print("\nFinal feature dataset saved successfully!")
print("File:", output_file)