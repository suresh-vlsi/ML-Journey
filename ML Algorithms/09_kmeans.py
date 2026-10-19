
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Generate data with three natural groups
X, true_labels = make_blobs(
    n_samples=600,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

# 2. Create the K-Means model
model = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=10,
    random_state=42
)

# 3. Discover clusters without providing labels
cluster_labels = model.fit_predict(X)

# 4. Extract the learned centroids
centroids = model.cluster_centers_

# 5. Evaluate the clustering
print(f"Number of clusters: {model.n_clusters}")
print(f"Number of iterations: {model.n_iter_}")
print(f"Inertia: {model.inertia_:.2f}")
print(f"Silhouette score: {silhouette_score(X, cluster_labels):.3f}")

# 6. Plot the discovered clusters
plt.figure(figsize=(10, 6))

plt.scatter(
    X[:, 0], X[:, 1],
    c=cluster_labels,
    cmap="viridis",
    s=35,
    alpha=0.8,
    label="Data points"
)

# 7. Plot the centroids
plt.scatter(
    centroids[:, 0], centroids[:, 1],
    marker="X",
    s=250,
    c="red",
    edgecolors="black",
    linewidths=1.5,
    label="Centroids"
)

plt.title("K-Means Clustering (K=3)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# 8. Plot the inertia for different values of K
k_values = range(1, 11)
inertias = []

for k in k_values:
    candidate = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )
    candidate.fit(X)
    inertias.append(candidate.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker="o")
plt.xticks(list(k_values))
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Choosing K")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
