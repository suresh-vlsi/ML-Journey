
import numpy as np

# =====================================================
# 1. INPUT AND TARGET
# =====================================================

x = np.array([[1.0],
              [2.0]])

y = np.array([[1.0]])

# =====================================================
# 2. INITIAL WEIGHTS AND BIASES
# =====================================================

W1 = np.array([
    [0.1, 0.2],
    [0.3, 0.4]
])

b1 = np.array([
    [0.1],
    [0.1]
])

W2 = np.array([[0.5, -0.4]])

b2 = np.array([[0.2]])

learning_rate = 0.1


# =====================================================
# 3. ACTIVATION AND LOSS FUNCTIONS
# =====================================================

def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y, y_hat):
    eps = 1e-12
    y_hat = np.clip(y_hat, eps, 1 - eps)

    return -np.mean(
        y * np.log(y_hat)
        + (1 - y) * np.log(1 - y_hat)
    )


# =====================================================
# 4. FORWARD PASS
# =====================================================

z1 = W1 @ x + b1
a1 = relu(z1)

z2 = W2 @ a1 + b2
y_hat = sigmoid(z2)

loss_before = binary_cross_entropy(y, y_hat)

print("FORWARD PASS")
print("Hidden pre-activations:\n", z1)
print("Hidden activations:\n", a1)
print("Output pre-activation:\n", z2)
print("Predicted probability:\n", y_hat)
print("Loss before update:", loss_before)


# =====================================================
# 5. BACKWARD PASS
# =====================================================

# Sigmoid output + binary cross-entropy
dz2 = y_hat - y

# Output-layer gradients
dW2 = dz2 @ a1.T
db2 = dz2.copy()

# Propagate the gradient into the hidden layer
da1 = W2.T @ dz2

# Derivative of ReLU
dz1 = da1 * (z1 > 0)

# Hidden-layer gradients
dW1 = dz1 @ x.T
db1 = dz1.copy()

print("\nBACKWARD PASS")
print("Output error dz2:\n", dz2)
print("Output weight gradient dW2:\n", dW2)
print("Output bias gradient db2:\n", db2)
print("Hidden error dz1:\n", dz1)
print("Hidden weight gradient dW1:\n", dW1)
print("Hidden bias gradient db1:\n", db1)


# =====================================================
# 6. UPDATE PARAMETERS
# =====================================================

W1 -= learning_rate * dW1
b1 -= learning_rate * db1

W2 -= learning_rate * dW2
b2 -= learning_rate * db2


# =====================================================
# 7. FORWARD PASS AFTER THE UPDATE
# =====================================================

z1_new = W1 @ x + b1
a1_new = relu(z1_new)

z2_new = W2 @ a1_new + b2
y_hat_new = sigmoid(z2_new)

loss_after = binary_cross_entropy(y, y_hat_new)

print("\nAFTER ONE GRADIENT DESCENT UPDATE")
print("Updated W1:\n", W1)
print("Updated b1:\n", b1)
print("Updated W2:\n", W2)
print("Updated b2:\n", b2)
print("New predicted probability:", y_hat_new.item())
print("New loss:", loss_after)
print("Loss decreased:", loss_after < loss_before)
