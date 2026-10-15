
import numpy as np

# ==========================================
# PART 1: MEAN SQUARED ERROR
# ==========================================

# Actual continuous values
y_true_reg = np.array([10.0, 20.0, 30.0])

# Predicted continuous values
y_pred_reg = np.array([12.0, 18.0, 33.0])

# Calculate squared errors
squared_errors = (y_true_reg - y_pred_reg) ** 2

# Average the squared errors
mse = np.mean(squared_errors)

print("MSE EXAMPLE")
print("Squared errors:", squared_errors)
print("MSE:", mse)


# ==========================================
# PART 2: BINARY CROSS-ENTROPY
# ==========================================

# Actual binary labels
y_true_bin = np.array([1.0, 0.0, 1.0, 0.0])

# Predicted probabilities for class 1
y_pred_bin = np.array([0.8, 0.2, 0.6, 0.1])

# Keep probabilities away from exactly 0 and 1
epsilon = 1e-15

p = np.clip(y_pred_bin, epsilon, 1 - epsilon)

# Calculate BCE for every example
bce_per_sample = -(
    y_true_bin * np.log(p)
    + (1 - y_true_bin) * np.log(1 - p)
)

# Average the losses
bce = np.mean(bce_per_sample)

print("\nBINARY CROSS-ENTROPY EXAMPLE")
print("Loss per sample:", bce_per_sample)
print("BCE:", bce)


# ==========================================
# PART 3: CATEGORICAL CROSS-ENTROPY
# ==========================================

# Three samples, three possible classes
y_true_cat = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Predicted class probabilities
y_pred_cat = np.array([
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.2, 0.2, 0.6]
])

# Avoid log(0)
p_cat = np.clip(y_pred_cat, epsilon, 1.0)

# Sum over classes for each sample
cce_per_sample = -np.sum(
    y_true_cat * np.log(p_cat),
    axis=1
)

# Average over samples
cce = np.mean(cce_per_sample)

print("\nCATEGORICAL CROSS-ENTROPY EXAMPLE")
print("Loss per sample:", cce_per_sample)
print("CCE:", cce)
