
import numpy as np

# 1. XOR dataset
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

# 2. Fixed random seed for reproducibility
rng = np.random.default_rng(42)

# 3. Network architecture: 2 inputs, 4 hidden neurons, 1 output
W1 = rng.normal(0, 0.5, (2, 4))
b1 = np.zeros((1, 4))

W2 = rng.normal(0, 0.5, (4, 1))
b2 = np.zeros((1, 1))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def tanh_derivative(z):
    return 1.0 - np.tanh(z) ** 2


def binary_cross_entropy(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1-y) * np.log(1-p))


def forward(X_batch):
    z1 = X_batch @ W1 + b1
    a1 = np.tanh(z1)

    z2 = a1 @ W2 + b2
    p = sigmoid(z2)

    return p, (z1, a1, p)


# 4. Training configuration
learning_rate = 0.1
epochs = 10000
batch_size = 4
n = len(X)

loss_history = []

# 5. Training loop
for epoch in range(epochs):

    # Shuffle examples at the start of each epoch
    order = rng.permutation(n)
    X_shuffled = X[order]
    y_shuffled = y[order]

    for start in range(0, n, batch_size):
        X_batch = X_shuffled[start:start + batch_size]
        y_batch = y_shuffled[start:start + batch_size]
        m = len(X_batch)

        # Forward pass
        p, (z1, a1, p) = forward(X_batch)

        # Backward pass
        dz2 = (p - y_batch) / m
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * tanh_derivative(z1)

        dW1 = X_batch.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Gradient-descent update
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    # Measure full-dataset loss after each epoch
    p_all, _ = forward(X)
    loss = binary_cross_entropy(y, p_all)
    loss_history.append(loss)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss = {loss:.6f}")

# 6. Final evaluation
p_all, _ = forward(X)
predictions = (p_all >= 0.5).astype(int)

print("\nFINAL XOR PREDICTIONS")
for i in range(n):
    print(
        f"Input={X[i].astype(int)} "
        f"Probability={p_all[i, 0]:.4f} "
        f"Prediction={predictions[i, 0]} "
        f"Target={int(y[i, 0])}"
    )

accuracy = np.mean(predictions == y)
print(f"\nAccuracy: {accuracy:.2%}")

# 7. Plot the training loss
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy loss")
plt.title("XOR Neural Network Training")
plt.grid(True)
plt.tight_layout()
plt.show()
