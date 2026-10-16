import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# --------------------------------
# Load dataset
# --------------------------------

print("Loading final feature dataset...")

df = pd.read_parquet(
    "data/final_features.parquet"
)

print("Dataset shape:", df.shape)


# Use sample for faster clustering
sample_size = 10000

if len(df) > sample_size:
    df = df.sample(
        n=sample_size,
        random_state=42
    )

print("Clustering dataset shape:", df.shape)


# --------------------------------
# Select clustering features
# --------------------------------

features = [
    "trip_distance",
    "passenger_count",
    "hour",
    "trip_duration"
]

X = df[features]


# --------------------------------
# Standardization
# --------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData standardization completed.")


# --------------------------------
# Elbow Method
# --------------------------------

print("\nCalculating Elbow Method...")

inertias = []

k_values = range(2, 7)

for k in k_values:

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertias.append(model.inertia_)

    print(
        "K =", k,
        "Inertia =", round(model.inertia_, 2)
    )


# Plot elbow curve

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")

plt.grid()

plt.savefig(
    "plots/elbow_method.png"
)

plt.show()


# --------------------------------
# Silhouette Score
# --------------------------------

print("\nCalculating Silhouette Scores...")

silhouette_scores = []

for k in k_values:

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)

    print(
        "K =", k,
        "Silhouette Score =",
        round(score, 4)
    )


# Plot silhouette scores

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for K-Means")

plt.grid()

plt.savefig(
    "plots/silhouette_scores.png"
)

plt.show()


# --------------------------------
# Final K-Means model
# --------------------------------

best_k = k_values[
    np.argmax(silhouette_scores)
]

print("\nSelected K:", best_k)


kmeans_model = KMeans(
    n_clusters=best_k,
    init="k-means++",
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans_model.fit_predict(
    X_scaled
)


# --------------------------------
# PCA
# --------------------------------

print("\nApplying PCA...")

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

print(
    "Explained variance:",
    pca.explained_variance_ratio_
)


# PCA visualization

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["Cluster"],
    s=10
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "K-Means Clusters using PCA"
)

plt.grid()

plt.savefig(
    "plots/kmeans_pca.png"
)

plt.show()


# --------------------------------
# Cluster summary
# --------------------------------

cluster_summary = df.groupby(
    "Cluster"
)[features].mean()

print("\nCluster Summary:")
print(cluster_summary)


# Save clustering results

df.to_parquet(
    "data/clustering_results.parquet",
    index=False
)

cluster_summary.to_csv(
    "results/cluster_summary.csv"
)


print("\n==============================")
print("M4 K-MEANS COMPLETED")
print("==============================")