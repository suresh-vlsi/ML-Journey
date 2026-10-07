import numpy as np


# ==========================================
# Dataset
# ==========================================

X = np.array([
    [1, 1],
    [1.5, 1],
    [2, 1.5],
    [2.5, 2],

    [7, 7],
    [8, 7],
    [8, 8],
    [9, 8]
])

y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
])


# ==========================================
# Parameters
# ==========================================

w = np.array([0.0, 0.0])
b = 0.0


learning_rate = 0.1
epochs = 1000


# ==========================================
# Functions
# ==========================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y, p):

    epsilon = 1e-15

    p = np.clip(p, epsilon, 1 - epsilon)

    loss = -(y * np.log(p) +
             (1 - y) * np.log(1 - p))

    return np.mean(loss)


# ==========================================
# Training
# ==========================================

for epoch in range(epochs):

    # Forward pass
    z = X @ w + b

    p = sigmoid(z)

    # Calculate loss
    loss = binary_cross_entropy(y, p)

    # Calculate gradient
    error = p - y

    dw = (X.T @ error) / len(X)

    db = np.mean(error)

    # Update parameters
    w = w - learning_rate * dw

    b = b - learning_rate * db

    # Print progress
    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}: "
            f"Loss = {loss:.6f}"
        )


# ==========================================
# Final prediction
# ==========================================

z = X @ w + b

p = sigmoid(z)

y_pred = (p >= 0.5).astype(int)


# ==========================================
# Results
# ==========================================

print("\nFinal weights:")
print(w)

print("\nFinal bias:")
print(b)

print("\nFinal probabilities:")
print(p)

print("\nPredictions:")
print(y_pred)

print("\nActual:")
print(y)