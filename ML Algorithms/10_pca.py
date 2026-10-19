
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Generate a dataset with 10 features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    random_state=42
)

# 2. Split the data before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Standardize features using training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Create PCA to retain two components
pca = PCA(n_components=2)

# 5. Learn components from training data
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 6. Display explained variance
print("Original number of features:", X_train.shape[1])
print("Number of principal components:", X_train_pca.shape[1])

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_.round(4))

print(
    "Total variance retained:",
    f"{pca.explained_variance_ratio_.sum():.2%}"
)

# 7. Reconstruct test data approximately
X_test_reconstructed = pca.inverse_transform(X_test_pca)

reconstruction_mse = np.mean(
    (X_test_scaled - X_test_reconstructed) ** 2
)

print(f"Reconstruction MSE: {reconstruction_mse:.4f}")

# 8. Plot the training data in principal-component space
plt.figure(figsize=(9, 6))

plt.scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    c=y_train,
    cmap="coolwarm",
    s=30,
    alpha=0.75
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: 10 Features Reduced to 2")
plt.colorbar(label="Class label")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# 9. Plot explained variance for all components
pca_full = PCA()
pca_full.fit(X_train_scaled)

individual_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(individual_variance)

components = np.arange(1, len(individual_variance) + 1)

plt.figure(figsize=(9, 5))

plt.bar(
    components,
    individual_variance,
    alpha=0.75,
    label="Individual explained variance"
)

plt.plot(
    components,
    cumulative_variance,
    marker="o",
    color="black",
    label="Cumulative explained variance"
)

plt.axhline(
    y=0.95,
    linestyle="--",
    label="95% variance reference"
)

plt.xticks(components)
plt.xlabel("Number of principal components")
plt.ylabel("Explained variance ratio")
plt.title("PCA Explained Variance")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
