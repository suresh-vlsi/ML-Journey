
# 07_loss_calculation.py
# Complete multiclass classification pipeline using NumPy.

import numpy as np

# Three samples, each with two input features
X = np.array([
    [2.0, 3.0],
    [1.0, 2.0],
    [3.0, 1.0]
])

# Weight matrix: 2 features x 3 classes
W = np.array([
    [1.0,  0.0, -1.0],
    [0.0,  1.0,  1.0]
])

# One bias for each class
b = np.array([0.0, 0.5, -0.5])

# Correct integer labels for the three samples
y = np.array([1, 2, 0])

# STEP 1: Calculate logits for every sample
logits = X @ W + b

# STEP 2: Calculate numerically stable softmax
shifted_logits = logits - np.max(
    logits, axis=1, keepdims=True
)

exp_values = np.exp(shifted_logits)

probabilities = exp_values / np.sum(
    exp_values, axis=1, keepdims=True
)

# STEP 3: Select the probability of each true class
correct_probs = probabilities[np.arange(len(y)), y]

# STEP 4: Calculate cross-entropy for each sample
epsilon = 1e-15
losses = -np.log(np.clip(correct_probs, epsilon, 1.0))

# STEP 5: Calculate average loss
mean_loss = np.mean(losses)

# STEP 6: Predict the class with the highest probability
predictions = np.argmax(probabilities, axis=1)

# STEP 7: Calculate accuracy
accuracy = np.mean(predictions == y)

print("Logits:\n", np.round(logits, 4))
print("\nProbabilities:\n", np.round(probabilities, 4))
print("\nTrue labels:", y)
print("Predictions:", predictions)
print("Correct-class probabilities:", np.round(correct_probs, 4))
print("Individual losses:", np.round(losses, 4))
print("Mean loss:", round(mean_loss, 4))
print("Accuracy:", round(accuracy * 100, 2), "%")
