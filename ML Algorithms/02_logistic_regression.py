
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Generate a two-class dataset
X, y = make_moons(
    n_samples=1000,
    noise=0.3,
    random_state=0
)

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=0,
    stratify=y
)

# 3. Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Predict classes for unseen test data
y_pred = model.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {accuracy:.2%}")
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# 6. Calculate class probabilities for the test samples
probabilities = model.predict_proba(X_test)
print("\nFirst 5 test probabilities:")
print(probabilities[:5].round(3))

# 7. Create a grid for visualizing the decision boundary
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(x2_min, x2_max, 300)
)

grid = np.column_stack((xx.ravel(), yy.ravel()))
prob_class1 = model.predict_proba(grid)[:, 1]
prob_class1 = prob_class1.reshape(xx.shape)

# 8. Plot the decision regions and training data
plt.figure(figsize=(10, 6))

plt.contourf(
    xx, yy, prob_class1,
    levels=[0, 0.5, 1],
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

plt.title(f"Logistic Regression (Test accuracy: {accuracy:.1%})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
