
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# Network layer dimensions
fan_in = 100
fan_out = 50

# 1. Small random initialization
W_small = rng.normal(
    0, 0.01, size=(fan_in, fan_out)
)

# 2. Xavier normal initialization
xavier_std = np.sqrt(2 / (fan_in + fan_out))
W_xavier = rng.normal(
    0, xavier_std, size=(fan_in, fan_out)
)

# 3. He normal initialization
he_std = np.sqrt(2 / fan_in)
W_he = rng.normal(
    0, he_std, size=(fan_in, fan_out)
)

# Print theoretical and observed standard deviations
print("INITIALIZATION COMPARISON")
print("-" * 45)

for name, weights, expected_std in [
    ("Small random", W_small, 0.01),
    ("Xavier normal", W_xavier, xavier_std),
    ("He normal", W_he, he_std)
]:
    print(f"\n{name}")
    print(f"Expected standard deviation: {expected_std:.5f}")
    print(f"Observed standard deviation: {np.std(weights):.5f}")
    print(f"Minimum weight: {np.min(weights):.5f}")
    print(f"Maximum weight: {np.max(weights):.5f}")

# Plot the weight distributions
plt.figure(figsize=(9, 5))

plt.hist(
    W_small.ravel(), bins=50,
    alpha=0.6, label="Small random"
)
plt.hist(
    W_xavier.ravel(), bins=50,
    alpha=0.6, label="Xavier normal"
)
plt.hist(
    W_he.ravel(), bins=50,
    alpha=0.6, label="He normal"
)

plt.xlabel("Weight value")
plt.ylabel("Frequency")
plt.title("Weight Initialization Distributions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
