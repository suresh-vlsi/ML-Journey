
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ==================================================
# STEP 1: CREATE A THREE-CLASS SPIRAL DATASET
# ==================================================

def make_three_spirals(samples_per_class=300, noise=0.06, seed=42):
    rng = np.random.default_rng(seed)

    X_parts = []
    y_parts = []

    for class_id in range(3):
        t = np.linspace(0, 1, samples_per_class)

        radius = 0.2 + 1.8 * t
        angle = 2 * np.pi * 1.25 * t + 2 * np.pi * class_id / 3

        x1 = radius * np.cos(angle)
        x2 = radius * np.sin(angle)

        points = np.column_stack((x1, x2))
        points += rng.normal(0, noise, points.shape)

        X_parts.append(points)
        y_parts.append(
            np.full(samples_per_class, class_id)
        )

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    order = rng.permutation(len(y))
    return X[order], y[order]


X, y = make_three_spirals()

print("X shape:", X.shape)
print("y shape:", y.shape)


# ==================================================
# STEP 2: SPLIT THE DATA
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==================================================
# STEP 3: STANDARDIZE THE FEATURES
# ==================================================

# Calculate statistics from training data only.
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# ==================================================
# STEP 4: ONE-HOT ENCODE THE CLASS LABELS
# ==================================================

num_classes = 3

Y_train = np.eye(num_classes)[y_train]

print("One-hot label shape:", Y_train.shape)


# ==================================================
# STEP 5: INITIALIZE THE NETWORK PARAMETERS
# ==================================================

rng = np.random.default_rng(1)

input_size = 2
hidden_size = 16
output_size = 3

# Input -> hidden layer
W1 = rng.normal(
    0, np.sqrt(2 / input_size),
    (input_size, hidden_size)
)
b1 = np.zeros((1, hidden_size))

# Hidden layer -> output layer
W2 = rng.normal(
    0, np.sqrt(2 / hidden_size),
    (hidden_size, output_size)
)
b2 = np.zeros((1, output_size))


# ==================================================
# STEP 6: DEFINE ACTIVATION AND LOSS FUNCTIONS
# ==================================================

def relu(z):
    return np.maximum(0, z)


def softmax(z):
    # Subtract row-wise maximum for numerical stability.
    z = z - np.max(z, axis=1, keepdims=True)

    exp_z = np.exp(z)

    return exp_z / np.sum(
        exp_z, axis=1, keepdims=True
    )


def cross_entropy(probabilities, labels):
    epsilon = 1e-12

    correct_class_probs = np.sum(
        labels * probabilities, axis=1
    )

    return -np.mean(
        np.log(correct_class_probs + epsilon)
    )


# ==================================================
# STEP 7: FORWARD PROPAGATION
# ==================================================

def forward(X):
    # First layer: weighted sum
    Z1 = X @ W1 + b1

    # First layer: nonlinear activation
    A1 = relu(Z1)

    # Output layer: class scores
    Z2 = A1 @ W2 + b2

    # Convert scores into probabilities
    P = softmax(Z2)

    return Z1, A1, Z2, P


# ==================================================
# STEP 8: TRAIN THE NETWORK
# ==================================================

learning_rate = 0.05
epochs = 2500
m = len(X_train)

loss_history = []

for epoch in range(epochs):

    # ----- Forward pass -----
    Z1, A1, Z2, P = forward(X_train)

    loss = cross_entropy(P, Y_train)
    loss_history.append(loss)

    # ----- Backward pass -----

    # Gradient of cross-entropy + softmax
    dZ2 = (P - Y_train) / m

    # Gradients for output-layer parameters
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # Send gradient back into the hidden layer
    dA1 = dZ2 @ W2.T

    # Gradient through ReLU
    dZ1 = dA1 * (Z1 > 0)

    # Gradients for hidden-layer parameters
    dW1 = X_train.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # ----- Gradient descent updates -----
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if (epoch + 1) % 250 == 0:
        print(
            f"Epoch {epoch + 1:4d} | "
            f"Loss: {loss:.4f}"
        )


# ==================================================
# STEP 9: EVALUATE ON TEST DATA
# ==================================================

_, _, _, test_probabilities = forward(X_test)

y_pred = np.argmax(test_probabilities, axis=1)

test_accuracy = accuracy_score(y_test, y_pred)

print("\nTest accuracy:", round(test_accuracy, 4))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==================================================
# STEP 10: PLOT THE TRAINING LOSS
# ==================================================

plt.figure(figsize=(7, 5))

plt.plot(loss_history)

plt.title("MLP Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("15_training_loss.png", dpi=200)
plt.show()


# ==================================================
# STEP 11: VISUALIZE THE DECISION BOUNDARIES
# ==================================================

x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.column_stack((xx.ravel(), yy.ravel()))

# Use training statistics to scale the grid.
grid_scaled = (grid - mean) / std

_, _, _, grid_probs = forward(grid_scaled)

grid_classes = np.argmax(grid_probs, axis=1)
regions = grid_classes.reshape(xx.shape)

plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    regions,
    levels=[-0.5, 0.5, 1.5, 2.5],
    alpha=0.3
)

for class_id in range(3):
    plt.scatter(
        X_test[y_test == class_id, 0],
        X_test[y_test == class_id, 1],
        label=f"Class {class_id}",
        s=20
    )

plt.title("MLP Decision Boundaries")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("15_mlp_decision_boundaries.png", dpi=200)
plt.show()
