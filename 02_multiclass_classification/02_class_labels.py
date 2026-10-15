
# 02_class_labels.py
import numpy as np

# Integer class labels for nine samples
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

num_classes = 3

# Convert integer labels to one-hot encoding
y_one_hot = np.eye(num_classes)[y]

print("Integer labels:")
print(y)

print("\nOne-hot encoded labels:")
print(y_one_hot)

print("\nInteger label shape:", y.shape)
print("One-hot label shape:", y_one_hot.shape)
