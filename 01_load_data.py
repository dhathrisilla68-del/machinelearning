import pandas as pd

file_path = "data/yellow_tripdata_2026-05.parquet"

print("Loading NYC TLC dataset...")

df = pd.read_parquet(file_path)

print("\nDataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df.describe())

print("\nData loading completed!")