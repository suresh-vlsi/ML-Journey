
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==================================================
# STEP 1: GENERATE THE THREE-CLASS SPIRAL DATASET
# ==================================================

def make_three_spirals(samples_per_class=300, noise=0.12, seed=42):
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
# STEP 2: TRAIN, VALIDATION, AND TEST SPLITS
# ==================================================

X_train_all, X_test, y_train_all, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_all, y_train_all,
    test_size=0.20,
    random_state=21,
    stratify=y_train_all
)

# Standardize using training data only.
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

num_classes = 3
Y_train = np.eye(num_classes)[y_train]
Y_val = np.eye(num_classes)[y_val]


# ==================================================
# STEP 3: BASIC FUNCTIONS
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
# STEP 4: FORWARD PASS
# ==================================================

def forward(X, params, training=False, dropout_rate=0.0, rng=None):
    W1, b1, W2, b2 = params

    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    dropout_mask = None

    if training and dropout_rate > 0:
        # Inverted dropout:
        # retained activations are scaled by 1/(1-rate).
        keep_probability = 1.0 - dropout_rate

        dropout_mask = (
            rng.random(A1.shape) < keep_probability
        ).astype(float)

        A1 = A1 * dropout_mask / keep_probability

    Z2 = A1 @ W2 + b2
    P = softmax(Z2)

    cache = (Z1, A1, dropout_mask)

    return P, cache


# ==================================================
# STEP 5: INITIALIZE PARAMETERS
# ==================================================

def initialize_network(hidden_size=32, seed=1):
    rng = np.random.default_rng(seed)

    W1 = rng.normal(
        0, np.sqrt(2 / 2),
        (2, hidden_size)
    )
    b1 = np.zeros((1, hidden_size))

    W2 = rng.normal(
        0, np.sqrt(2 / hidden_size),
        (hidden_size, 3)
    )
    b2 = np.zeros((1, 3))

    return [W1, b1, W2, b2]


# ==================================================
# STEP 6: TRAIN ONE MODEL
# ==================================================

def train_model(
    name,
    l2_lambda=0.0,
    dropout_rate=0.0,
    epochs=1500,
    learning_rate=0.03,
    seed=1
):
    params = initialize_network(hidden_size=32, seed=seed)
    rng = np.random.default_rng(seed + 100)

    W1, b1, W2, b2 = params
    m = len(X_train)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_params = None

    for epoch in range(epochs):

        # ----- Forward pass with dropout during training -----
        P, cache = forward(
            X_train,
            params,
            training=True,
            dropout_rate=dropout_rate,
            rng=rng
        )

        Z1, A1, dropout_mask = cache

        # Data loss plus L2 penalty on weights.
        data_loss = cross_entropy(P, Y_train)

        regularization_loss = (
            l2_lambda / (2 * m)
        ) * (
            np.sum(W1 ** 2) + np.sum(W2 ** 2)
        )

        # ----- Backpropagation -----
        dZ2 = (P - Y_train) / m

        dW2 = A1.T @ dZ2 + (l2_lambda / m) * W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T

        # Backpropagate through dropout using the same mask.
        if dropout_mask is not None:
            keep_probability = 1.0 - dropout_rate
            dA1 = (
                dA1 * dropout_mask / keep_probability
            )

        # Backpropagate through ReLU.
        dZ1 = dA1 * (Z1 > 0)

        dW1 = X_train.T @ dZ1 + (l2_lambda / m) * W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # ----- Gradient descent -----
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        # ----- Evaluate without dropout -----
        # Validation uses the complete network.
        train_prob, _ = forward(X_train, params)
        val_prob, _ = forward(X_val, params)

        train_data_loss = cross_entropy(train_prob, Y_train)
        val_data_loss = cross_entropy(val_prob, Y_val)

        train_total_loss = train_data_loss + (
            l2_lambda / (2 * m)
        ) * (
            np.sum(W1 ** 2) + np.sum(W2 ** 2)
        )

        val_total_loss = val_data_loss + (
            l2_lambda / (2 * m)
        ) * (
            np.sum(W1 ** 2) + np.sum(W2 ** 2)
        )

        train_pred = np.argmax(train_prob, axis=1)
        val_pred = np.argmax(val_prob, axis=1)

        history["train_loss"].append(train_total_loss)
        history["val_loss"].append(val_total_loss)

        history["train_accuracy"].append(
            accuracy_score(y_train, train_pred)
        )
        history["val_accuracy"].append(
            accuracy_score(y_val, val_pred)
        )

        # Save the parameters with the lowest validation loss.
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_epoch = epoch + 1
            best_params = [p.copy() for p in params]

    print(f"\n{name}")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", round(best_val_loss, 4))
    print(
        "Best validation accuracy:",
        round(max(history["val_accuracy"]), 4)
    )

    return {
        "name": name,
        "history": history,
        "params": best_params,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss
    }


# ==================================================
# STEP 7: TRAIN THREE CONFIGURATIONS
# ==================================================

baseline = train_model(
    "Baseline MLP",
    l2_lambda=0.0,
    dropout_rate=0.0
)

l2_model = train_model(
    "L2 Regularization",
    l2_lambda=0.01,
    dropout_rate=0.0
)

dropout_model = train_model(
    "Dropout",
    l2_lambda=0.0,
    dropout_rate=0.25
)

models = [baseline, l2_model, dropout_model]


# ==================================================
# STEP 8: FINAL TEST EVALUATION
# ==================================================

print("\n--- Final Test Results ---")

for model in models:
    probabilities, _ = forward(
        X_test,
        model["params"],
        training=False
    )

    predictions = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, predictions)

    print(f"{model['name']}: {accuracy:.4f}")


# ==================================================
# STEP 9: PLOT VALIDATION LOSSES
# ==================================================

plt.figure(figsize=(9, 5))

for model in models:
    plt.plot(
        model["history"]["val_loss"],
        label=model["name"]
    )

plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.title("Regularization Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("17_validation_loss.png", dpi=200)
plt.show()


# ==================================================
# STEP 10: PLOT VALIDATION ACCURACIES
# ==================================================

plt.figure(figsize=(9, 5))

for model in models:
    plt.plot(
        model["history"]["val_accuracy"],
        label=model["name"]
    )

plt.xlabel("Epoch")
plt.ylabel("Validation accuracy")
plt.title("Validation Accuracy Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("17_validation_accuracy.png", dpi=200)
plt.show()
