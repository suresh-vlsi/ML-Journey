
# 04_logits.py
# Calculate logits for a three-class classifier.

import numpy as np

# One sample with two input features
X = np.array([2.0, 3.0])

# Weight matrix: 2 features x 3 classes
W = np.array([
    [1.0,  0.0, -1.0],
    [0.0,  1.0,  1.0]
])

# One bias per class
b = np.array([0.0, 0.5, -0.5])

# Calculate all three logits
logits = X @ W + b

print("Input features:", X)
print("Weight matrix:\n", W)
print("Bias vector:", b)
print("Logits:", logits)

# Find the class with the highest score
predicted_class = np.argmax(logits)

print("Predicted class:", predicted_class)
