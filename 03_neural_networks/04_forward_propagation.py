
import numpy as np

# ==========================================
# STEP 1: Define the input features
# ==========================================

x = np.array([
    [1.0],
    [2.0]
])

# ==========================================
# STEP 2: Define hidden-layer parameters
# ==========================================

W1 = np.array([
    [0.5, -1.0],
    [1.0,  0.5]
])

b1 = np.array([
    [0.0],
    [-0.5]
])

# ==========================================
# STEP 3: Define output-layer parameters
# ==========================================

W2 = np.array([
    [1.0, -1.0]
])

b2 = np.array([
    [0.5]
])

# ==========================================
# STEP 4: Define activation functions
# ==========================================

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# ==========================================
# STEP 5: Forward propagation
# ==========================================

# Hidden layer: weighted sum
z1 = W1 @ x + b1

# Hidden layer: activation
h = relu(z1)

# Output layer: weighted sum
z2 = W2 @ h + b2

# Output layer: sigmoid activation
y_hat = sigmoid(z2)

# ==========================================
# STEP 6: Print intermediate calculations
# ==========================================

print("Input x:")
print(x)

print("\nHidden weighted sum z1:")
print(z1)

print("\nHidden activation h:")
print(h)

print("\nOutput weighted sum z2:")
print(z2)

print("\nFinal output y_hat:")
print(y_hat)

print("\nPredicted class:")
print(int(y_hat[0, 0] >= 0.5))
