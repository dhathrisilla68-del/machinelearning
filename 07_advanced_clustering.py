import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest

print("Loading final feature dataset...")

df = pd.read_parquet(
    "data/final_features.parquet"
)

print("Original dataset shape:", df.shape)

# Use 10000 samples
sample_size = 10000

if len(df) > sample_size:
    df = df.sample(
        n=sample_size,
        random_state=42
    )

print("Clustering dataset shape:", df.shape)

# Features
features = [
    "trip_distance",
    "passenger_count",
    "hour",
    "trip_duration"
]

X = df[features]

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData standardization completed.")


# ==================================================
# 1. MINI-BATCH K-MEANS
# ==================================================

print("\n==============================")
print("MINI-BATCH K-MEANS")
print("==============================")

mini_kmeans = MiniBatchKMeans(
    n_clusters=2,
    batch_size=256,
    random_state=42,
    n_init=10
)

mini_labels = mini_kmeans.fit_predict(X_scaled)

mini_silhouette = silhouette_score(
    X_scaled,
    mini_labels
)

print("Number of clusters:", 2)
print(
    "Silhouette Score:",
    round(mini_silhouette, 4)
)


# ==================================================
# 2. HIERARCHICAL CLUSTERING
# ==================================================

print("\n==============================")
print("HIERARCHICAL CLUSTERING")
print("==============================")

hierarchical = AgglomerativeClustering(
    n_clusters=2,
    linkage="ward"
)

hierarchical_labels = hierarchical.fit_predict(
    X_scaled
)

hierarchical_silhouette = silhouette_score(
    X_scaled,
    hierarchical_labels
)

print("Linkage method: Ward")
print("Number of clusters:", 2)
print(
    "Silhouette Score:",
    round(hierarchical_silhouette, 4)
)


# ==================================================
# 3. DBSCAN
# ==================================================

print("\n==============================")
print("DBSCAN CLUSTERING")
print("==============================")

dbscan = DBSCAN(
    eps=0.8,
    min_samples=10
)

dbscan_labels = dbscan.fit_predict(
    X_scaled
)

unique_labels = set(dbscan_labels)

number_of_clusters = len(
    unique_labels - {-1}
)

number_of_noise = list(
    dbscan_labels
).count(-1)

print(
    "Number of clusters:",
    number_of_clusters
)

print(
    "Noise / anomaly points:",
    number_of_noise
)

if number_of_clusters > 1:

    dbscan_silhouette = silhouette_score(
        X_scaled,
        dbscan_labels
    )

    print(
        "Silhouette Score:",
        round(dbscan_silhouette, 4)
    )


# ==================================================
# 4. ANOMALY DETECTION
# ==================================================

print("\n==============================")
print("ANOMALY DETECTION")
print("==============================")

isolation_forest = IsolationForest(
    contamination=0.01,
    random_state=42
)

anomaly_labels = isolation_forest.fit_predict(
    X_scaled
)

df["Anomaly"] = anomaly_labels

normal_count = (anomaly_labels == 1).sum()
anomaly_count = (anomaly_labels == -1).sum()

print(
    "Normal records:",
    normal_count
)

print(
    "Anomalous records:",
    anomaly_count
)


# ==================================================
# SAVE RESULTS
# ==================================================

df["MiniBatch_Cluster"] = mini_labels
df["Hierarchical_Cluster"] = hierarchical_labels
df["DBSCAN_Cluster"] = dbscan_labels

output_file = (
    "data/advanced_clustering_results.parquet"
)

df.to_parquet(
    output_file,
    index=False
)

print("\nAdvanced clustering results saved!")

print("File:", output_file)


# ==================================================
# CLUSTER SUMMARY
# ==================================================

summary = df.groupby(
    "MiniBatch_Cluster"
)[features].mean()

print("\nMini-Batch K-Means Cluster Summary:")
print(summary)

summary.to_csv(
    "results/mini_batch_cluster_summary.csv"
)


# ==================================================
# FINAL MESSAGE
# ==================================================

print("\n==============================")
print("M4 ADVANCED CLUSTERING COMPLETED")
print("==============================")