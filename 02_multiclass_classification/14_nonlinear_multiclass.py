
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# STEP 1: Generate a three-class spiral dataset
# --------------------------------------------------

def make_three_spirals(n_samples=900, noise=0.06, seed=42):
    rng = np.random.default_rng(seed)

    samples_per_class = n_samples // 3
    X_parts = []
    y_parts = []

    for class_id in range(3):
        t = np.linspace(0, 1, samples_per_class)

        # Radius gradually increases along each spiral.
        radius = 0.2 + 1.8 * t

        # Each class starts at a different angle.
        angle = 2 * np.pi * 1.25 * t + 2 * np.pi * class_id / 3

        x1 = radius * np.cos(angle)
        x2 = radius * np.sin(angle)

        points = np.column_stack((x1, x2))

        # Add random noise to make classification harder.
        points += rng.normal(
            loc=0.0,
            scale=noise,
            size=points.shape
        )

        X_parts.append(points)
        y_parts.append(
            np.full(samples_per_class, class_id)
        )

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Shuffle all samples together.
    order = rng.permutation(len(y))
    return X[order], y[order]


X, y = make_three_spirals()

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)
print("Classes:", np.unique(y))


# --------------------------------------------------
# STEP 2: Visualize the original dataset
# --------------------------------------------------

plt.figure(figsize=(7, 6))

for class_id in range(3):
    plt.scatter(
        X[y == class_id, 0],
        X[y == class_id, 1],
        label=f"Class {class_id}",
        s=18
    )

plt.title("Three-Class Spiral Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("14_spiral_dataset.png", dpi=200)
plt.show()


# --------------------------------------------------
# STEP 3: Split into training and testing data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# STEP 4: Build two different classifiers
# --------------------------------------------------

linear_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

nonlinear_model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=10, gamma="scale")
)


# --------------------------------------------------
# STEP 5: Train both models
# --------------------------------------------------

linear_model.fit(X_train, y_train)
nonlinear_model.fit(X_train, y_train)


# --------------------------------------------------
# STEP 6: Evaluate both models
# --------------------------------------------------

linear_predictions = linear_model.predict(X_test)
nonlinear_predictions = nonlinear_model.predict(X_test)

linear_accuracy = accuracy_score(
    y_test, linear_predictions
)

nonlinear_accuracy = accuracy_score(
    y_test, nonlinear_predictions
)

print("\n--- Test Accuracy ---")
print(f"Logistic Regression: {linear_accuracy:.4f}")
print(f"RBF SVM:             {nonlinear_accuracy:.4f}")

print("\n--- RBF SVM Classification Report ---")
print(
    classification_report(
        y_test,
        nonlinear_predictions,
        zero_division=0
    )
)


# --------------------------------------------------
# STEP 7: Create a grid for decision-boundary plots
# --------------------------------------------------

x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid_points = np.column_stack((
    xx.ravel(),
    yy.ravel()
))


# --------------------------------------------------
# STEP 8: Plot each model's decision boundaries
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

models = [
    (linear_model, "Logistic Regression (Linear)"),
    (nonlinear_model, "SVM with RBF Kernel (Nonlinear)")
]

for ax, (model, title) in zip(axes, models):

    # Predict the class at every grid location.
    grid_predictions = model.predict(grid_points)

    # Reshape predictions into the grid's 2D shape.
    regions = grid_predictions.reshape(xx.shape)

    ax.contourf(
        xx,
        yy,
        regions,
        alpha=0.25,
        levels=[-0.5, 0.5, 1.5, 2.5]
    )

    # Overlay the actual data points.
    for class_id in range(3):
        ax.scatter(
            X_test[y_test == class_id, 0],
            X_test[y_test == class_id, 1],
            label=f"Class {class_id}",
            s=18
        )

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("14_decision_boundaries.png", dpi=200)
plt.show()
