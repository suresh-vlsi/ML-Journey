
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# 3. Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=0,
    n_jobs=-1
)

# 4. Train all trees in the forest
model.fit(X_train, y_train)

# 5. Predict on unseen test data
y_pred = model.predict(X_test)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Number of trees: {len(model.estimators_)}")
print(f"Test accuracy: {accuracy:.2%}")
print(f"Training accuracy: {model.score(X_train, y_train):.2%}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Create a grid for the decision-boundary plot
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(x2_min, x2_max, 300)
)

grid = np.column_stack((xx.ravel(), yy.ravel()))

# 8. Predict class probabilities across the grid
prob_class1 = model.predict_proba(grid)[:, 1]
prob_class1 = prob_class1.reshape(xx.shape)

# 9. Plot the ensemble decision boundary
plt.figure(figsize=(10, 6))

plt.contourf(
    xx, yy, prob_class1,
    levels=[0, 0.5, 1],
    cmap="coolwarm",
    alpha=0.25
)

plt.contour(
    xx, yy, prob_class1,
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

plt.title(f"Random Forest (300 trees, accuracy={accuracy:.1%})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# 10. Plot feature importance
importance = model.feature_importances_
feature_names = ["Feature 1", "Feature 2"]

plt.figure(figsize=(7, 4))
plt.bar(feature_names, importance)
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()
