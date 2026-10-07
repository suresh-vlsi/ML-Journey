import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# DATA
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
# ACTIVATIONS
# ============================================================

def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ============================================================
# LOSS
# ============================================================

def binary_cross_entropy(y, p):

    eps = 1e-15

    p = np.clip(p, eps, 1 - eps)

    return -np.mean(
        y * np.log(p) +
        (1 - y) * np.log(1 - p)
    )


# ============================================================
# INITIALIZATION
# ============================================================

np.random.seed(42)

# 2 → 8
W1 = np.random.randn(2, 8) * np.sqrt(2 / 2)
b1 = np.zeros((1, 8))

# 8 → 8
W2 = np.random.randn(8, 8) * np.sqrt(2 / 8)
b2 = np.zeros((1, 8))

# 8 → 1
W3 = np.random.randn(8, 1) * 0.1
b3 = np.zeros((1, 1))


# ============================================================
# TRAINING
# ============================================================

learning_rate = 0.1
epochs = 5000

batch_size = 4

lambda_reg = 0.001

loss_history = []


for epoch in range(epochs):

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    z1 = X @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    a2 = relu(z2)

    z3 = a2 @ W3 + b3
    y_hat = sigmoid(z3)


    # --------------------------------------------------------
    # Data loss
    # --------------------------------------------------------

    data_loss = binary_cross_entropy(y, y_hat)


    # --------------------------------------------------------
    # L2 regularization
    # --------------------------------------------------------

    reg_loss = (
        lambda_reg / 2
        * (
            np.sum(W1 ** 2) +
            np.sum(W2 ** 2) +
            np.sum(W3 ** 2)
        )
    )

    loss = data_loss + reg_loss

    loss_history.append(loss)


    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    m = len(X)

    dz3 = y_hat - y

    dW3 = (a2.T @ dz3) / m + lambda_reg * W3

    db3 = np.mean(dz3, axis=0, keepdims=True)


    dz2 = (
        (dz3 @ W3.T)
        * relu_derivative(z2)
    )

    dW2 = (a1.T @ dz2) / m + lambda_reg * W2

    db2 = np.mean(dz2, axis=0, keepdims=True)


    dz1 = (
        (dz2 @ W2.T)
        * relu_derivative(z1)
    )

    dW1 = (X.T @ dz1) / m + lambda_reg * W1

    db1 = np.mean(dz1, axis=0, keepdims=True)


    # --------------------------------------------------------
    # Gradient descent
    # --------------------------------------------------------

    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1


# ============================================================
# FINAL FORWARD PASS
# ============================================================

a1 = relu(X @ W1 + b1)

a2 = relu(a1 @ W2 + b2)

y_hat = sigmoid(a2 @ W3 + b3)

y_pred = (y_hat >= 0.5).astype(int)


# ============================================================
# RESULTS
# ============================================================

print("\nFINAL RESULTS")
print("=" * 60)

for i in range(len(X)):

    print(
        f"Input = {X[i]} | "
        f"Probability = {y_hat[i, 0]:.4f} | "
        f"Prediction = {y_pred[i, 0]} | "
        f"Actual = {int(y[i, 0])}"
    )


accuracy = np.mean(y_pred == y)

print("\nAccuracy:", accuracy)


# ============================================================
# LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(loss_history)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("MLP Training")

plt.grid(True)

plt.show()