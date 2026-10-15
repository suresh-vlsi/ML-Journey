
# 01_dataset.py
# Generate a simple dataset for three-class classification.

import numpy as np
import matplotlib.pyplot as plt

# Reproducible random numbers
np.random.seed(42)

# Create three groups of data points.
class_0 = np.random.randn(100, 2) + np.array([0, 0])
class_1 = np.random.randn(100, 2) + np.array([4, 0])
class_2 = np.random.randn(100, 2) + np.array([2, 4])

# Combine the input features.
X = np.vstack((class_0, class_1, class_2))

# Create integer labels.
y = np.array(
    [0] * 100 +
    [1] * 100 +
    [2] * 100
)

# Display dataset dimensions.
print("Input shape:", X.shape)
print("Label shape:", y.shape)
print("First five samples:\n", X[:5])
print("First five labels:", y[:5])

# Visualize the dataset.
plt.figure(figsize=(8, 6))

plt.scatter(class_0[:, 0], class_0[:, 1], label="Class 0")
plt.scatter(class_1[:, 0], class_1[:, 1], label="Class 1")
plt.scatter(class_2[:, 0], class_2[:, 1], label="Class 2")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Three-Class Classification Dataset")
plt.legend()
plt.grid(True)
plt.show()
