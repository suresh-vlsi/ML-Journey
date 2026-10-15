
# 08_gradients.py
# Calculate gradients for softmax + cross-entropy.

import numpy as np

# One sample, two input features
X = np.array([2.0, -1.0])

# Three predicted probabilities
probabilities = np.array([0.20, 0.50, 0.30])

# Correct class is Class 1
y = np.array([0.0, 1.0, 0.0])

# Step 1: Gradient of loss with respect to logits
dz = probabilities - y

# Step 2: Gradient of loss with respect to weights
dW = np.outer(X, dz)

# Step 3: Gradient of loss with respect to biases
db = dz.copy()

print("Predicted probabilities:", probabilities)
print("True one-hot label:", y)

print("\nGradient with respect to logits:")
print(dz)

print("\nGradient with respect to weights:")
print(dW)

print("\nGradient with respect to biases:")
print(db)

print("\nSum of logit gradients:", np.sum(dz))
