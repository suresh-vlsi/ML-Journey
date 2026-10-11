import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. DATA: XOR
# ============================================================

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)


# ============================================================
# 2. ACTIVATIONS AND LOSS
# ============================================================

def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def sigmoid(z):
    # Clip to avoid overflow in exp for extreme values.
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, probabilities):
    eps = 1e-15
    probabilities = np.clip(probabilities, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(probabilities)
        + (1 - y_true) * np.log(1 - probabilities)
    )


# ============================================================
# 3. INITIALIZATION
# ============================================================

np.random.seed(42)

input_size = 2
hidden1_size = 8
hidden2_size = 8
output_size = 1

W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2 / input_size)
b1 = np.zeros((1, hidden1_size))

W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2 / hidden1_size)
b2 = np.zeros((1, hidden2_size))

W3 = np.random.randn(hidden2_size, output_size) * 0.1
b3 = np.zeros((1, output_size))


# ============================================================
# 4. FORWARD PASS
# ============================================================

def forward(X_input):
    z1 = X_input @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    a2 = relu(z2)

    z3 = a2 @ W3 + b3
    probabilities = sigmoid(z3)

    # Return intermediate values for backpropagation.
    cache = (z1, a1, z2, a2, z3, probabilities)
    return probabilities, cache


# ============================================================
# 5. TRAINING
# ============================================================

learning_rate = 0.1
epochs = 5000
lambda_reg = 0.001
m = len(X)

loss_history = []

for epoch in range(epochs):
    # Forward pass
    y_hat, cache = forward(X)
    z1, a1, z2, a2, z3, y_hat = cache

    # Loss = data loss + L2 weight penalty
    data_loss = binary_cross_entropy(y, y_hat)
    reg_loss = (lambda_reg / 2) * (
        np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2)
    )
    loss_history.append(data_loss + reg_loss)

    # Backpropagation: sigmoid + binary cross-entropy
    dz3 = y_hat - y
    dW3 = (a2.T @ dz3) / m + lambda_reg * W3
    db3 = np.mean(dz3, axis=0, keepdims=True)

    dz2 = (dz3 @ W3.T) * relu_derivative(z2)
    dW2 = (a1.T @ dz2) / m + lambda_reg * W2
    db2 = np.mean(dz2, axis=0, keepdims=True)

    dz1 = (dz2 @ W2.T) * relu_derivative(z1)
    dW1 = (X.T @ dz1) / m + lambda_reg * W1
    db1 = np.mean(dz1, axis=0, keepdims=True)

    # Gradient descent
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1


# ============================================================
# 6. FINAL PREDICTIONS
# ============================================================

y_hat, _ = forward(X)
y_pred = (y_hat >= 0.5).astype(int)

print("\nFINAL XOR RESULTS")
print("=" * 65)
for i in range(len(X)):
    print(
        f"Input = {X[i].astype(int)} | "
        f"Probability = {y_hat[i, 0]:.4f} | "
        f"Prediction = {y_pred[i, 0]} | "
        f"Actual = {int(y[i, 0])}"
    )

accuracy = np.mean(y_pred == y)
print(f"\nAccuracy: {accuracy:.2%}")
print(f"Final loss: {loss_history[-1]:.6f}")


# ============================================================
# 7. CREATE A MESH GRID FOR THE DECISION BOUNDARY
# ============================================================

# Expand beyond [0, 1] so the shape of the boundary is visible.
x1_min, x1_max = -0.5, 1.5
x2_min, x2_max = -0.5, 1.5

grid_step = 0.01
xx1, xx2 = np.meshgrid(
    np.arange(x1_min, x1_max + grid_step, grid_step),
    np.arange(x2_min, x2_max + grid_step, grid_step)
)

# Each grid location is one candidate input [x1, x2].
grid_points = np.column_stack((xx1.ravel(), xx2.ravel()))

# Evaluate the trained network at every grid location.
grid_probabilities, _ = forward(grid_points)
Z = grid_probabilities.reshape(xx1.shape)


# ============================================================
# 8. PLOT THE NONLINEAR DECISION BOUNDARY
# ============================================================

plt.figure(figsize=(8, 7))

# Show the model's predicted probability over the plane.
contour = plt.contourf(
    xx1, xx2, Z,
    levels=np.linspace(0, 1, 21),
    cmap="coolwarm",
    alpha=0.75
)
plt.colorbar(contour, label="Predicted probability P(y = 1)")

# The 0.5 contour is the classification decision boundary.
boundary = plt.contour(
    xx1, xx2, Z,
    levels=[0.5],
    colors="black",
    linewidths=2.5
)
plt.clabel(boundary, fmt={0.5: "Boundary: p = 0.5"}, inline=True, fontsize=9)

# Plot the four actual XOR samples.
plt.scatter(
    X[:, 0], X[:, 1],
    c=y.ravel(),
    cmap="coolwarm",
    vmin=0, vmax=1,
    edgecolors="black",
    s=180,
    zorder=3
)

for i, (x1, x2) in enumerate(X):
    plt.annotate(
        f"y={int(y[i, 0])}",
        (x1, x2),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=10,
        weight="bold"
    )

plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.xlabel("Input x1")
plt.ylabel("Input x2")
plt.title("MLP Learned Nonlinear Decision Boundary for XOR")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# ============================================================
# 9. PLOT TRAINING LOSS
# ============================================================

plt.figure(figsize=(8, 4.5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Training Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
