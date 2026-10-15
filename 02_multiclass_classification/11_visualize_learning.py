
# 11_visualize_learning.py
# Train a three-class classifier and visualize its decision boundaries.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --------------------------------------------------
# STEP 1: GENERATE THE DATASET
# --------------------------------------------------

class_0 = np.random.randn(100, 2) + [0, 0]
class_1 = np.random.randn(100, 2) + [4, 0]
class_2 = np.random.randn(100, 2) + [2, 4]

X = np.vstack((class_0, class_1, class_2))
y = np.array([0] * 100 + [1] * 100 + [2] * 100)

# Shuffle and split
indices = np.random.permutation(len(y))
X = X[indices]
y = y[indices]

split = int(0.8 * len(y))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --------------------------------------------------
# STEP 2: INITIALIZE THE MODEL
# --------------------------------------------------

num_features = 2
num_classes = 3

W = np.random.randn(num_features, num_classes) * 0.01
b = np.zeros(num_classes)

learning_rate = 0.1
epochs = 500
N = len(y_train)
rows = np.arange(N)


def softmax(logits):
    shifted = logits - np.max(
        logits, axis=1, keepdims=True
    )
    exp_values = np.exp(shifted)
    return exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )


# --------------------------------------------------
# STEP 3: TRAIN THE CLASSIFIER
# --------------------------------------------------

loss_history = []

for epoch in range(epochs):
    logits = X_train @ W + b
    probabilities = softmax(logits)

    correct_probs = probabilities[rows, y_train]
    loss = -np.mean(
        np.log(np.clip(correct_probs, 1e-15, 1.0))
    )
    loss_history.append(loss)

    dZ = probabilities.copy()
    dZ[rows, y_train] -= 1
    dZ /= N

    dW = X_train.T @ dZ
    db = np.sum(dZ, axis=0)

    W -= learning_rate * dW
    b -= learning_rate * db

# --------------------------------------------------
# STEP 4: EVALUATE THE TEST DATA
# --------------------------------------------------

test_probabilities = softmax(X_test @ W + b)
test_predictions = np.argmax(test_probabilities, axis=1)

accuracy = np.mean(test_predictions == y_test)

print("Learned weights:\n", np.round(W, 3))
print("Learned biases:", np.round(b, 3))
print("Final recorded training loss:", round(loss_history[-1], 4))
print("Test accuracy:", round(accuracy * 100, 2), "%")

# --------------------------------------------------
# STEP 5: CREATE A GRID OF INPUT POINTS
# --------------------------------------------------

x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

grid_x1, grid_x2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(x2_min, x2_max, 300)
)

# Convert every grid point into a two-feature sample
grid_points = np.column_stack((
    grid_x1.ravel(),
    grid_x2.ravel()
))

# Predict a class for every grid point
grid_probabilities = softmax(grid_points @ W + b)
grid_predictions = np.argmax(grid_probabilities, axis=1)

# Reshape predictions to match the grid
regions = grid_predictions.reshape(grid_x1.shape)

# --------------------------------------------------
# STEP 6: PLOT DECISION REGIONS AND DATA
# --------------------------------------------------

plt.figure(figsize=(10, 7))

plt.contourf(
    grid_x1, grid_x2, regions,
    levels=[-0.5, 0.5, 1.5, 2.5],
    alpha=0.25
)

# Draw boundaries between predicted classes
plt.contour(
    grid_x1, grid_x2, regions,
    levels=[0.5, 1.5],
    linewidths=1.5
)

# Plot training samples
for class_id in range(num_classes):
    mask = y_train == class_id
    plt.scatter(
        X_train[mask, 0],
        X_train[mask, 1],
        label=f"Class {class_id} (train)",
        alpha=0.65,
        s=25
    )

# Mark misclassified test samples with black crosses
wrong = test_predictions != y_test

plt.scatter(
    X_test[wrong, 0],
    X_test[wrong, 1],
    marker="x",
    s=100,
    linewidths=2,
    label="Misclassified test samples"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Learned Multiclass Decision Boundaries")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("11_decision_boundaries.png", dpi=150)
plt.show()

# --------------------------------------------------
# STEP 7: PLOT THE LOSS CURVE
# --------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("11_loss_curve.png", dpi=150)
plt.show()

print("Saved 11_decision_boundaries.png")
print("Saved 11_loss_curve.png")
