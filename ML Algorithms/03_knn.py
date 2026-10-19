
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Generate a two-class dataset
X, y = make_moons(
    n_samples=1000,
    noise=0.3,
    random_state=0
)

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=0,
    stratify=y
)

# 3. Create the KNN model with feature scaling
k = 15

model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=k)
)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Predict on unseen test samples
y_pred = model.predict(X_test)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Number of neighbors (k): {k}")
print(f"Test accuracy: {accuracy:.2%}")
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Create a grid for plotting the decision boundary
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(x2_min, x2_max, 300)
)

grid = np.column_stack((xx.ravel(), yy.ravel()))

# 8. Predict the class at every grid location
Z = model.predict(grid).reshape(xx.shape)

# 9. Plot decision regions and samples
plt.figure(figsize=(10, 6))

plt.contourf(
    xx, yy, Z,
    levels=[-0.5, 0.5, 1.5],
    cmap="coolwarm",
    alpha=0.25
)

plt.contour(
    xx, yy, Z,
    levels=[0.5],
    colors="black",
    linewidths=2
)

plt.scatter(
    X_train[:, 0], X_train[:, 1],
    c=y_train, cmap="coolwarm",
    edgecolors="black", s=25,
    label="Training data"
)

plt.scatter(
    X_test[:, 0], X_test[:, 1],
    c=y_test, cmap="coolwarm",
    marker="x", s=35,
    label="Test data"
)

plt.title(f"K-Nearest Neighbors (k={k}, accuracy={accuracy:.1%})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
