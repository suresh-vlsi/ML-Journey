
# 05_softmax.py
# Implement softmax from scratch using NumPy.

import numpy as np

# Logits calculated in Lesson 4
logits = np.array([2.0, 3.5, 0.5])

# Step 1: Subtract the largest logit for numerical stability
shifted_logits = logits - np.max(logits)

# Step 2: Calculate exponentials
exp_values = np.exp(shifted_logits)

# Step 3: Calculate the sum of exponentials
total = np.sum(exp_values)

# Step 4: Normalize to obtain probabilities
probabilities = exp_values / total

# Step 5: Predict the class with the highest probability
predicted_class = np.argmax(probabilities)

print("Original logits:", logits)
print("Shifted logits:", shifted_logits)
print("Exponentials:", exp_values)
print("Sum of exponentials:", total)
print("Probabilities:", probabilities)
print("Probability sum:", np.sum(probabilities))
print("Predicted class:", predicted_class)
