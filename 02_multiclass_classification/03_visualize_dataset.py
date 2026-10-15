
# 03_visualize_dataset.py

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate three classes
class_0 = np.random.randn(100, 2) + [0, 0]
class_1 = np.random.randn(100, 2) + [4, 0]
class_2 = np.random.randn(100, 2) + [2, 4]

# Combine all samples
X = np.vstack((class_0, class_1, class_2))

# Create class labels
y = np.array([0] * 100 + [1] * 100 + [2] * 100)

# Plot the dataset
plt.figure(figsize=(9, 7))

for class_id in range(3):
    mask = y == class_id

    plt.scatter(
        X[mask, 0],
        X[mask, 1],
        label=f"Class {class_id}",
        alpha=0.7
    )


print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)
print("Samples per class:", np.bincount(y))
print("Saved plot as 03_dataset.png")

plt.xlabel("Feature 1 (x1)")
plt.ylabel("Feature 2 (x2)")
plt.title("Multiclass Classification Dataset")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("03_dataset.png", dpi=150)
plt.show()

