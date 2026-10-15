
import numpy as np

# ==========================================
# STEP 1: Define the training example
# ==========================================

x = 2.0
target = 1.0

# Initial parameters
w = 0.5
b = 0.0

# Learning rate
learning_rate = 0.1


# ==========================================
# STEP 2: Define sigmoid activation
# ==========================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ==========================================
# STEP 3: Forward propagation
# ==========================================

z = w * x + b
prediction = sigmoid(z)

# Squared error for this one sample
loss = (target - prediction) ** 2


# ==========================================
# STEP 4: Apply the chain rule
# ==========================================

dL_dprediction = 2 * (prediction - target)

dprediction_dz = prediction * (1 - prediction)

dz_dw = x

# Multiply the local derivatives
dL_dw = (
    dL_dprediction
    * dprediction_dz
    * dz_dw
)


# ==========================================
# STEP 5: Calculate a numerical gradient
# ==========================================

epsilon = 1e-5

def calculate_loss(weight):
    z_value = weight * x + b
    predicted_value = sigmoid(z_value)
    return (target - predicted_value) ** 2

loss_plus = calculate_loss(w + epsilon)
loss_minus = calculate_loss(w - epsilon)

numerical_gradient = (
    loss_plus - loss_minus
) / (2 * epsilon)


# ==========================================
# STEP 6: Update the weight
# ==========================================

new_w = w - learning_rate * dL_dw


# ==========================================
# STEP 7: Display results
# ==========================================

print("Weighted sum:", z)
print("Prediction:", prediction)
print("Loss:", loss)

print("dL/dprediction:", dL_dprediction)
print("dprediction/dz:", dprediction_dz)
print("dz/dw:", dz_dw)

print("Analytical gradient:", dL_dw)
print("Numerical gradient:", numerical_gradient)

print("Old weight:", w)
print("Updated weight:", new_w)
