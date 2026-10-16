import pandas as pd

# Load raw dataset
file_path = "data/yellow_tripdata_2026-05.parquet"

print("Loading raw dataset...")

df = pd.read_parquet(file_path)

print("Raw dataset shape:", df.shape)

# Convert date and time columns
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

# Calculate trip duration in minutes
df["trip_duration"] = (
    df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
).dt.total_seconds() / 60

# Remove invalid values
df = df[
    (df["trip_duration"] > 1) &
    (df["trip_duration"] <= 180) &
    (df["trip_distance"] > 0) &
    (df["trip_distance"] <= 100) &
    (df["passenger_count"] > 0) &
    (df["passenger_count"] <= 8)
]

# Remove missing values
required_columns = [
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "trip_duration"
]

df = df.dropna(subset=required_columns)

print("Cleaned dataset shape:", df.shape)

print("\nMissing values after cleaning:")
print(df[required_columns].isnull().sum())

print("\nFirst 5 cleaned records:")
print(df.head())

# Save cleaned dataset
output_file = "data/cleaned_taxi_data.parquet"

df.to_parquet(output_file, index=False)

print("\nCleaned dataset saved successfully!")
print("File:", output_file)