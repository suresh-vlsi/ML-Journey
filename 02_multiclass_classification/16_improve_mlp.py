
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==================================================
# STEP 1: GENERATE THREE-CLASS SPIRAL DATA
# ==================================================

def make_three_spirals(samples_per_class=300, noise=0.06, seed=42):
    rng = np.random.default_rng(seed)

    X_parts = []
    y_parts = []

    for class_id in range(3):
        t = np.linspace(0, 1, samples_per_class)

        radius = 0.2 + 1.8 * t
        angle = 2 * np.pi * 1.25 * t + 2 * np.pi * class_id / 3

        points = np.column_stack((
            radius * np.cos(angle),
            radius * np.sin(angle)
        ))

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


# ==================================================
# STEP 2: CREATE TRAIN, VALIDATION, AND TEST SETS
# ==================================================

X_train_all, X_test, y_train_all, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_all,
    y_train_all,
    test_size=0.20,
    random_state=21,
    stratify=y_train_all
)

# 64% training, 16% validation, 20% testing.
# Fit scaling statistics on training data only.
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

num_classes = 3

Y_train = np.eye(num_classes)[y_train]
Y_val = np.eye(num_classes)[y_val]


# ==================================================
# STEP 3: ACTIVATION, SOFTMAX, AND LOSS
# ==================================================

def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy(probabilities, one_hot_labels):
    eps = 1e-12

    correct_probabilities = np.sum(
        probabilities * one_hot_labels,
        axis=1
    )

    return -np.mean(np.log(correct_probabilities + eps))


# ==================================================
# STEP 4: TRAIN ONE MLP CONFIGURATION
# ==================================================

def train_mlp(
    hidden_size,
    learning_rate,
    epochs=1000,
    seed=1
):
    rng = np.random.default_rng(seed)

    input_size = 2
    output_size = 3

    # He-style initialization for ReLU.
    W1 = rng.normal(
        0,
        np.sqrt(2 / input_size),
        (input_size, hidden_size)
    )
    b1 = np.zeros((1, hidden_size))

    W2 = rng.normal(
        0,
        np.sqrt(2 / hidden_size),
        (hidden_size, output_size)
    )
    b2 = np.zeros((1, output_size))

    n = len(X_train)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_parameters = None

    for epoch in range(epochs):

        # ----- Forward pass on training data -----
        Z1 = X_train @ W1 + b1
        A1 = relu(Z1)

        Z2 = A1 @ W2 + b2
        P = softmax(Z2)

        loss = cross_entropy(P, Y_train)

        # ----- Backpropagation -----
        dZ2 = (P - Y_train) / n

        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * (Z1 > 0)

        dW1 = X_train.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # ----- Gradient descent -----
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        # ----- Evaluate after the parameter update -----
        train_prob = softmax(
            relu(X_train @ W1 + b1) @ W2 + b2
        )

        val_prob = softmax(
            relu(X_val @ W1 + b1) @ W2 + b2
        )

        train_loss = cross_entropy(train_prob, Y_train)
        val_loss = cross_entropy(val_prob, Y_val)

        train_pred = np.argmax(train_prob, axis=1)
        val_pred = np.argmax(val_prob, axis=1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_accuracy"].append(
            accuracy_score(y_train, train_pred)
        )
        history["val_accuracy"].append(
            accuracy_score(y_val, val_pred)
        )

        # Keep the parameters with the lowest validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            best_parameters = (
                W1.copy(),
                b1.copy(),
                W2.copy(),
                b2.copy()
            )

    return {
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "parameters": best_parameters
    }


# ==================================================
# STEP 5: EXPERIMENT WITH HIDDEN SIZE AND LEARNING RATE
# ==================================================

hidden_sizes = [8, 16, 32]
learning_rates = [0.01, 0.05, 0.20]

results = []

for hidden_size in hidden_sizes:
    for lr in learning_rates:

        result = train_mlp(
            hidden_size=hidden_size,
            learning_rate=lr,
            epochs=1000,
            seed=1
        )

        results.append(result)

        h = result["history"]

        print(
            f"Hidden={hidden_size:2d}, "
            f"LR={lr:.2f}, "
            f"Best epoch={result['best_epoch']:4d}, "
            f"Val accuracy="
            f"{max(h['val_accuracy']):.4f}, "
            f"Best val loss={result['best_val_loss']:.4f}"
        )


# ==================================================
# STEP 6: SELECT THE BEST VALIDATION CONFIGURATION
# ==================================================

best_result = min(
    results,
    key=lambda result: result["best_val_loss"]
)

print("\nSelected configuration:")
print("Hidden neurons:", best_result["hidden_size"])
print("Learning rate:", best_result["learning_rate"])
print("Best epoch:", best_result["best_epoch"])
print("Best validation loss:", round(best_result["best_val_loss"], 4))


# ==================================================
# STEP 7: EVALUATE ON THE HELD-OUT TEST SET
# ==================================================

W1, b1, W2, b2 = best_result["parameters"]

test_probabilities = softmax(
    relu(X_test @ W1 + b1) @ W2 + b2
)

test_predictions = np.argmax(test_probabilities, axis=1)

test_accuracy = accuracy_score(y_test, test_predictions)

print("\nFinal test accuracy:", round(test_accuracy, 4))


# ==================================================
# STEP 8: PLOT TRAINING AND VALIDATION LOSS
# ==================================================

history = best_result["history"]

plt.figure(figsize=(8, 5))
plt.plot(history["train_loss"], label="Training loss")
plt.plot(history["val_loss"], label="Validation loss")

plt.axvline(
    best_result["best_epoch"] - 1,
    linestyle="--",
    label="Selected epoch"
)

plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("16_loss_comparison.png", dpi=200)
plt.show()


# ==================================================
# STEP 9: COMPARE VALIDATION ACCURACY
# ==================================================

plt.figure(figsize=(8, 5))

labels = [
    f"H={r['hidden_size']}, LR={r['learning_rate']}"
    for r in results
]

accuracies = [
    max(r["history"]["val_accuracy"])
    for r in results
]

plt.bar(labels, accuracies)
plt.ylabel("Best validation accuracy")
plt.xlabel("Network configuration")
plt.title("MLP Hyperparameter Comparison")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("16_hyperparameter_comparison.png", dpi=200)
plt.show()
