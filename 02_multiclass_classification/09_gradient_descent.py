
# 09_gradient_descent.py
# Perform one gradient descent update for multiclass classification.

import numpy as np

# One training sample with two features
X = np.array([2.0, -1.0])

# True class: Class 1
y = np.array([0.0, 1.0, 0.0])

# Start with zero weights and biases
W = np.zeros((2, 3))
b = np.zeros(3)

# Learning rate
learning_rate = 0.1


def softmax(logits):
    # Stable softmax
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def cross_entropy(probabilities, target):
    # Loss for a one-hot target
    correct_probability = np.sum(probabilities * target)
    return -np.log(max(correct_probability, 1e-15))


# ----- BEFORE UPDATE -----

logits_before = X @ W + b
probs_before = softmax(logits_before)
loss_before = cross_entropy(probs_before, y)

# ----- CALCULATE GRADIENTS -----

# Gradient of loss with respect to logits
dz = probs_before - y

# Gradients of weights and biases
dW = np.outer(X, dz)
db = dz.copy()

# ----- UPDATE PARAMETERS -----

W = W - learning_rate * dW
b = b - learning_rate * db

# ----- AFTER UPDATE -----

logits_after = X @ W + b
probs_after = softmax(logits_after)
loss_after = cross_entropy(probs_after, y)

print("Initial probabilities:", probs_before)
print("Initial loss:", round(loss_before, 4))

print("\nWeight gradient:\n", dW)
print("Bias gradient:", db)

print("\nUpdated weights:\n", W)
print("Updated biases:", b)

print("\nNew logits:", logits_after)
print("New probabilities:", probs_after)
print("New loss:", round(loss_after, 4))
print("Loss decreased:", loss_after < loss_before)
