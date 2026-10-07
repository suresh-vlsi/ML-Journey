import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. XOR DATASET
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
# 2. ACTIVATION FUNCTIONS
# ============================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y, p):

    epsilon = 1e-15

    p = np.clip(p, epsilon, 1 - epsilon)

    return -np.mean(
        y * np.log(p) +
        (1 - y) * np.log(1 - p)
    )


# ============================================================
# 3. INITIALIZE PARAMETERS
# ============================================================

np.random.seed(42)

# 2 inputs → 3 hidden neurons
W1 = np.random.randn(2, 3) * 0.5

b1 = np.zeros((1, 3))

# 3 hidden neurons → 1 output
W2 = np.random.randn(3, 1) * 0.5

b2 = np.zeros((1, 1))


# ============================================================
# 4. TRAINING SETTINGS
# ============================================================

learning_rate = 1.0

epochs = 10000

loss_history = []


# ============================================================
# 5. TRAINING LOOP
# ============================================================

for epoch in range(epochs):

    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    z1 = X @ W1 + b1

    a1 = sigmoid(z1)

    z2 = a1 @ W2 + b2

    y_hat = sigmoid(z2)


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    loss = binary_cross_entropy(y, y_hat)

    loss_history.append(loss)


    # --------------------------------------------------------
    # BACKPROPAGATION
    # --------------------------------------------------------

    # Output layer
    dz2 = y_hat - y

    dW2 = (a1.T @ dz2) / len(X)

    db2 = np.mean(dz2, axis=0, keepdims=True)


    # Hidden layer
    dz1 = (
        (dz2 @ W2.T)
        * a1
        * (1 - a1)
    )

    dW1 = (X.T @ dz1) / len(X)

    db1 = np.mean(dz1, axis=0, keepdims=True)


    # --------------------------------------------------------
    # PARAMETER UPDATE
    # --------------------------------------------------------

    W2 -= learning_rate * dW2

    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1

    b1 -= learning_rate * db1


# ============================================================
# 6. FINAL PREDICTIONS
# ============================================================

z1 = X @ W1 + b1

a1 = sigmoid(z1)

z2 = a1 @ W2 + b2

y_hat = sigmoid(z2)

y_pred = (y_hat >= 0.5).astype(int)


# ============================================================
# 7. RESULTS
# ============================================================

print("\nFINAL RESULTS")
print("=" * 50)

for i in range(len(X)):

    print(
        f"Input: {X[i]} | "
        f"Probability: {y_hat[i,0]:.4f} | "
        f"Prediction: {y_pred[i,0]} | "
        f"Actual: {int(y[i,0])}"
    )


accuracy = np.mean(y_pred == y)

print("\nAccuracy:", accuracy)


# ============================================================
# 8. LOSS CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(loss_history)

plt.xlabel("Epoch")

plt.ylabel("Binary Cross-Entropy Loss")

plt.title("Neural Network Learning on XOR")

plt.grid(True)

plt.show()