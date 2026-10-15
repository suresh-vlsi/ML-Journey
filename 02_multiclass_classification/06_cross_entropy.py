
# 06_cross_entropy.py
# Calculate categorical cross-entropy from scratch.

import numpy as np

# Three samples, three classes each
probabilities = np.array([
    [0.80, 0.10, 0.10],
    [0.20, 0.60, 0.20],
    [0.05, 0.05, 0.90]
])

# Correct class for each sample
y = np.array([0, 1, 2])

# Select the probability assigned to each correct class
correct_probs = probabilities[np.arange(len(y)), y]

# Avoid log(0) in more general cases
epsilon = 1e-15
correct_probs = np.clip(correct_probs, epsilon, 1.0)

# Calculate loss for each sample
losses = -np.log(correct_probs)

# Calculate average loss
mean_loss = np.mean(losses)

print("Correct class probabilities:", correct_probs)
print("Individual losses:", losses)
print("Mean cross-entropy loss:", mean_loss)
