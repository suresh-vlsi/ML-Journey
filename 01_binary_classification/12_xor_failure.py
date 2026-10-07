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
])

y = np.array([
    0,
    1,
    1,
    0
])


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

w = np.array([0.1, -0.1])

b = 0.0

learning_rate = 0.1

epochs = 5000

loss_history = []


for epoch in range(epochs):

    # Forward pass
    z = X @ w + b

    p = sigmoid(z)

    # Loss
    loss = binary_cross_entropy(y, p)

    loss_history.append(loss)

    # Gradient
    error = p - y

    dw = (X.T @ error) / len(X)

    db = np.mean(error)

    # Update
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
# 5. ACCURACY
# ============================================================

accuracy = np.mean(y_pred == y)

print("\nAccuracy:", accuracy)


# ============================================================
# 6. LOSS CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(loss_history)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("XOR: Training Loss")

plt.grid(True)

plt.show()


# ============================================================
# 7. DECISION BOUNDARY
# ============================================================

plt.figure(figsize=(7, 7))


plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    label="Class 0",
    s=100
)


plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    label="Class 1",
    s=100
)


# Boundary:
#
# w1*x1 + w2*x2 + b = 0
#
# x2 = -(w1*x1+b)/w2

x1_values = np.linspace(-0.2, 1.2, 100)

if abs(w[1]) > 1e-10:

    x2_values = (
        -(w[0] * x1_values + b)
        / w[1]
    )

    plt.plot(
        x1_values,
        x2_values,
        label="Decision Boundary"
    )


plt.xlim(-0.2, 1.2)

plt.ylim(-0.2, 1.2)

plt.xlabel("Feature 1")

plt.ylabel("Feature 2")

plt.title("XOR: Linear Classifier Failure")

plt.legend()

plt.grid(True)

plt.show()

print("\nFinal weights:", w)
print("Final bias:", b)

print("\nXOR Predictions")
print("================")

for i in range(len(X)):
    print(
        f"Input: {X[i]} | "
        f"Probability: {p[i]:.4f} | "
        f"Prediction: {y_pred[i]} | "
        f"Actual: {y[i]}"
    )