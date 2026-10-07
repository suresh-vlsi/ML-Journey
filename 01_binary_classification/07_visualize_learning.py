import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. DATASET
# ============================================================

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

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])


# ============================================================
# 2. FUNCTIONS
# ============================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y, p):

    epsilon = 1e-15

    p = np.clip(p, epsilon, 1 - epsilon)

    loss = -(y * np.log(p) +
             (1 - y) * np.log(1 - p))

    return np.mean(loss)


# ============================================================
# 3. TRAINING
# ============================================================

w = np.array([0.0, 0.0])
b = 0.0

learning_rate = 0.1
epochs = 1000

loss_history = []


for epoch in range(epochs):

    # Forward pass
    z = X @ w + b

    p = sigmoid(z)

    # Calculate loss
    loss = binary_cross_entropy(y, p)

    loss_history.append(loss)

    # Gradient
    error = p - y

    dw = (X.T @ error) / len(X)

    db = np.mean(error)

    # Update parameters
    w = w - learning_rate * dw

    b = b - learning_rate * db


# ============================================================
# 4. FINAL PREDICTIONS
# ============================================================

z = X @ w + b

p = sigmoid(z)

y_pred = (p >= 0.5).astype(int)


print("Final weights:", w)
print("Final bias:", b)

print("\nProbabilities:")
print(p)

print("\nPredictions:")
print(y_pred)

print("\nActual:")
print(y)


# ============================================================
# 5. PLOT 1 — LOSS CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")

plt.title("Learning Process: Loss vs Epoch")

plt.grid(True)

plt.tight_layout()


# ============================================================
# 6. PLOT 2 — DECISION BOUNDARY
# ============================================================

plt.figure(figsize=(8, 6))

# Class 0
plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    label="Class 0"
)

# Class 1
plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    label="Class 1"
)


# Generate x1 values
x1_values = np.linspace(0, 10, 100)


# Decision boundary:
#
# w1*x1 + w2*x2 + b = 0
#
# Therefore:
#
# x2 = -(w1*x1 + b) / w2

x2_values = -(w[0] * x1_values + b) / w[1]


plt.plot(
    x1_values,
    x2_values,
    label="Decision Boundary"
)


plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.title("Learned Decision Boundary")

plt.legend()

plt.grid(True)

plt.tight_layout()


# ============================================================
# 7. SHOW BOTH FIGURES
# ============================================================

plt.show()