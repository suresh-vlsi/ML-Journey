import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. DATASET
# ============================================================

X = np.array([
    [1, 1],
    [1.5, 1],
    [2, 1.5],
    [2.5, 2],
    [7, 7],
    [8, 7],
    [8, 8],
    [9, 8]
])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])


# ============================================================
# 2. FUNCTIONS
# ============================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y, p):

    epsilon = 1e-15

    p = np.clip(p, epsilon, 1 - epsilon)

    loss = -(y * np.log(p) +
             (1 - y) * np.log(1 - p))

    return np.mean(loss)


# ============================================================
# 3. TRAINING SETTINGS
# ============================================================

learning_rate = 0.1
epochs = 1000

w = np.array([0.0, 0.0])
b = 0.0


# Epochs that we want to observe
snapshot_epochs = [0, 10, 50, 100, 500, 1000]

snapshots = {}


# ============================================================
# 4. TRAINING
# ============================================================

for epoch in range(epochs + 1):

    # ----------------------------------------
    # Forward pass
    # ----------------------------------------

    z = X @ w + b

    p = sigmoid(z)

    loss = binary_cross_entropy(y, p)


    # ----------------------------------------
    # Save model state
    # ----------------------------------------

    if epoch in snapshot_epochs:

        snapshots[epoch] = {
            "w": w.copy(),
            "b": b,
            "loss": loss
        }


    # ----------------------------------------
    # Gradient
    # ----------------------------------------

    error = p - y

    dw = (X.T @ error) / len(X)

    db = np.mean(error)


    # ----------------------------------------
    # Gradient descent
    # ----------------------------------------

    w = w - learning_rate * dw

    b = b - learning_rate * db


# ============================================================
# 5. PRINT SNAPSHOT INFORMATION
# ============================================================

print("\nLEARNING PROCESS")
print("=" * 50)

for epoch in snapshot_epochs:

    w_snapshot = snapshots[epoch]["w"]
    b_snapshot = snapshots[epoch]["b"]
    loss_snapshot = snapshots[epoch]["loss"]

    print(
        f"Epoch {epoch:4d} | "
        f"w = {w_snapshot} | "
        f"b = {b_snapshot:.4f} | "
        f"Loss = {loss_snapshot:.6f}"
    )


# ============================================================
# 6. CREATE VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(14, 8)
)


# Generate x-axis values
x1_values = np.linspace(0, 10, 100)


for ax, epoch in zip(axes.ravel(), snapshot_epochs):

    # --------------------------------------------------------
    # Get saved model
    # --------------------------------------------------------

    w_snapshot = snapshots[epoch]["w"]
    b_snapshot = snapshots[epoch]["b"]

    loss_snapshot = snapshots[epoch]["loss"]


    # --------------------------------------------------------
    # Plot Class 0
    # --------------------------------------------------------

    ax.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        label="Class 0"
    )


    # --------------------------------------------------------
    # Plot Class 1
    # --------------------------------------------------------

    ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        label="Class 1"
    )


    # --------------------------------------------------------
    # Decision boundary
    #
    # w1*x1 + w2*x2 + b = 0
    #
    # x2 = -(w1*x1 + b)/w2
    # --------------------------------------------------------

    if abs(w_snapshot[1]) > 1e-10:

        x2_values = (
            -(w_snapshot[0] * x1_values + b_snapshot)
            / w_snapshot[1]
        )

        ax.plot(
            x1_values,
            x2_values,
            label="Decision Boundary"
        )


    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    ax.set_title(
        f"Epoch {epoch}\nLoss = {loss_snapshot:.4f}"
    )

    ax.grid(True)

    ax.legend()


plt.suptitle(
    "How the Decision Boundary Learns",
    fontsize=16
)

plt.tight_layout()

plt.show()