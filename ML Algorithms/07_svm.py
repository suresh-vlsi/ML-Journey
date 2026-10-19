
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
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

# 3. Create a scaled RBF-kernel SVM
model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, gamma="scale")
)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Predict on unseen test data
y_pred = model.predict(X_test)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {accuracy:.2%}")
print(f"Training accuracy: {model.score(X_train, y_train):.2%}")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Access the trained SVM inside the pipeline
svm = model.named_steps["svc"]
scaler = model.named_steps["standardscaler"]

print(f"\nNumber of support vectors: {svm.n_support_}")
print(f"Total support vectors: {len(svm.support_)}")

# Convert support vectors back to original feature coordinates
support_vectors = scaler.inverse_transform(svm.support_vectors_)

# 8. Create a grid for the decision-boundary plot
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(x2_min, x2_max, 300)
)

grid = np.column_stack((xx.ravel(), yy.ravel()))

# 9. Calculate the decision function over the grid
decision_values = model.decision_function(grid)
decision_values = decision_values.reshape(xx.shape)

# 10. Plot decision regions, boundary, and support vectors
plt.figure(figsize=(10, 6))

plt.contourf(
    xx, yy, decision_values,
    levels=[-100, 0, 100],
    cmap="coolwarm",
    alpha=0.25
)

# Decision boundary: decision function = 0
plt.contour(
    xx, yy, decision_values,
    levels=[0],
    colors="black",
    linewidths=2
)

# Margin contours for the SVM decision function
plt.contour(
    xx, yy, decision_values,
    levels=[-1, 1],
    colors="gray",
    linestyles="--",
    linewidths=1.5
)

# Training and test samples
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

# Highlight support vectors
plt.scatter(
    support_vectors[:, 0], support_vectors[:, 1],
    s=100, facecolors="none", edgecolors="green",
    linewidths=1.5,
    label="Support vectors"
)

plt.title(f"RBF SVM (Test accuracy: {accuracy:.1%})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
