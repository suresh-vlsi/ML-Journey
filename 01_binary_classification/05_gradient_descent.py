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
# Initial parameters
# ==========================================

w = np.array([1.0, 1.0])
b = -5.0


# ==========================================
# Learning rate
# ==========================================

learning_rate = 0.1


# ==========================================
# Sigmoid
# ==========================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ==========================================
# Loss
# ==========================================

def binary_cross_entropy(y, p):

    epsilon = 1e-15

    p = np.clip(p, epsilon, 1 - epsilon)

    loss = -(y * np.log(p) +
             (1 - y) * np.log(1 - p))

    return np.mean(loss)


# ==========================================
# Forward pass
# ==========================================

z = X @ w + b

p = sigmoid(z)


# ==========================================
# Loss before update
# ==========================================

loss_before = binary_cross_entropy(y, p)


# ==========================================
# Gradient
# ==========================================

error = p - y

dw = (X.T @ error) / len(X)

db = np.mean(error)


# ==========================================
# Gradient descent update
# ==========================================

w = w - learning_rate * dw

b = b - learning_rate * db


# ==========================================
# Forward pass again
# ==========================================

z_new = X @ w + b

p_new = sigmoid(z_new)


# ==========================================
# Loss after update
# ==========================================

loss_after = binary_cross_entropy(y, p_new)


# ==========================================
# Results
# ==========================================

print("Gradient dw:")
print(dw)

print("\nGradient db:")
print(db)

print("\nUpdated weights:")
print(w)

print("\nUpdated bias:")
print(b)

print("\nLoss before update:")
print(loss_before)

print("\nLoss after update:")
print(loss_after)