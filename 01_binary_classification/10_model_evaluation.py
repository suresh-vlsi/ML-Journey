import numpy as np


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

y = np.array([
    0, 0, 0, 0,
    1, 1, 1, 1
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
# 3. TRAIN MODEL
# ============================================================

w = np.array([0.0, 0.0])

b = 0.0

learning_rate = 0.1

epochs = 1000


for epoch in range(epochs):

    # Forward pass
    z = X @ w + b

    p = sigmoid(z)

    # Gradient
    error = p - y

    dw = (X.T @ error) / len(X)

    db = np.mean(error)

    # Update
    w = w - learning_rate * dw

    b = b - learning_rate * db


# ============================================================
# 4. PREDICTION
# ============================================================

z = X @ w + b

probabilities = sigmoid(z)

y_pred = (probabilities >= 0.5).astype(int)


# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

TP = np.sum((y == 1) & (y_pred == 1))

TN = np.sum((y == 0) & (y_pred == 0))

FP = np.sum((y == 0) & (y_pred == 1))

FN = np.sum((y == 1) & (y_pred == 0))


# ============================================================
# 6. METRICS
# ============================================================

accuracy = (TP + TN) / len(y)

precision = TP / (TP + FP)

recall = TP / (TP + FN)

f1 = 2 * precision * recall / (precision + recall)


# ============================================================
# 7. RESULTS
# ============================================================

print("Final Model")
print("===========")

print("Weights:", w)

print("Bias:", b)

print("\nProbabilities:")
print(probabilities)

print("\nPredictions:")
print(y_pred)

print("\nActual:")
print(y)

print("\nConfusion Matrix")
print("================")
print(
    np.array([
        [TN, FP],
        [FN, TP]
    ])
)

print("\nMetrics")
print("=======")

print("Accuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1 Score :", f1)