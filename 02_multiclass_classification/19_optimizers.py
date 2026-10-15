
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==================================================
# STEP 1: GENERATE THREE-CLASS SPIRAL DATA
# ==================================================

def make_three_spirals(samples_per_class=300, noise=0.08, seed=42):
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []

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
        y_parts.append(np.full(samples_per_class, class_id))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    order = rng.permutation(len(y))
    return X[order], y[order]


X, y = make_three_spirals()

X_train_all, X_test, y_train_all, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_all, y_train_all,
    test_size=0.20, random_state=21, stratify=y_train_all
)

# Standardize using training data only.
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

Y_train = np.eye(3)[y_train]
Y_val = np.eye(3)[y_val]


# ==================================================
# STEP 2: ACTIVATION, SOFTMAX, AND LOSS
# ==================================================

def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def cross_entropy(p, one_hot_labels):
    eps = 1e-12
    correct_p = np.sum(p * one_hot_labels, axis=1)
    return -np.mean(np.log(correct_p + eps))


# ==================================================
# STEP 3: INITIALIZE THE NETWORK
# ==================================================

def initialize_network(seed=1, hidden_size=32):
    rng = np.random.default_rng(seed)

    W1 = rng.normal(
        0, np.sqrt(2 / 2), (2, hidden_size)
    )
    b1 = np.zeros((1, hidden_size))

    W2 = rng.normal(
        0, np.sqrt(2 / hidden_size), (hidden_size, 3)
    )
    b2 = np.zeros((1, 3))

    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }


def forward(X, params):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)

    Z2 = A1 @ params["W2"] + params["b2"]
    P = softmax(Z2)

    cache = (Z1, A1)
    return P, cache


# ==================================================
# STEP 4: CALCULATE GRADIENTS
# ==================================================

def calculate_gradients(X, Y, params):
    m = len(X)

    P, (Z1, A1) = forward(X, params)

    dZ2 = (P - Y) / m

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    gradients = {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2
    }

    return gradients


# ==================================================
# STEP 5: DEFINE THE OPTIMIZERS
# ==================================================

def optimizer_update(
    params,
    gradients,
    state,
    optimizer_name,
    learning_rate,
    step
):
    beta = 0.9
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    for key in params:
        g = gradients[key]

        if optimizer_name == "SGD":
            params[key] -= learning_rate * g

        elif optimizer_name == "Momentum":
            state[key] = beta * state[key] + g
            params[key] -= learning_rate * state[key]

        elif optimizer_name == "Adam":
            state["m"][key] = (
                beta1 * state["m"][key]
                + (1 - beta1) * g
            )

            state["v"][key] = (
                beta2 * state["v"][key]
                + (1 - beta2) * (g ** 2)
            )

            m_hat = state["m"][key] / (1 - beta1 ** step)
            v_hat = state["v"][key] / (1 - beta2 ** step)

            params[key] -= (
                learning_rate * m_hat
                / (np.sqrt(v_hat) + epsilon)
            )

        else:
            raise ValueError("Unknown optimizer")


# ==================================================
# STEP 6: TRAIN A MODEL
# ==================================================

def train_model(optimizer_name, epochs=1500):
    # Identical starting parameters for each optimizer.
    params = initialize_network(seed=1)

    if optimizer_name == "SGD":
        learning_rate = 0.1
    elif optimizer_name == "Momentum":
        learning_rate = 0.05
    else:
        learning_rate = 0.01

    # Initialize optimizer-specific state.
    if optimizer_name in ("SGD", "Momentum"):
        state = {
            key: np.zeros_like(value)
            for key, value in params.items()
        }
    else:
        state = {
            "m": {
                key: np.zeros_like(value)
                for key, value in params.items()
            },
            "v": {
                key: np.zeros_like(value)
                for key, value in params.items()
            }
        }

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_loss = float("inf")
    best_params = None
    best_epoch = 0

    for epoch in range(epochs):
        # Forward pass and gradients.
        gradients = calculate_gradients(
            X_train, Y_train, params
        )

        # Update parameters.
        optimizer_update(
            params,
            gradients,
            state,
            optimizer_name,
            learning_rate,
            epoch + 1
        )

        # Evaluate updated model.
        train_p, _ = forward(X_train, params)
        val_p, _ = forward(X_val, params)

        train_loss = cross_entropy(train_p, Y_train)
        val_loss = cross_entropy(val_p, Y_val)

        val_pred = np.argmax(val_p, axis=1)
        val_accuracy = accuracy_score(y_val, val_pred)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_params = {
                key: value.copy()
                for key, value in params.items()
            }

    print(f"\n{optimizer_name}")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", round(best_val_loss, 4))
    print(
        "Best validation accuracy:",
        round(max(history["val_accuracy"]), 4)
    )

    return {
        "name": optimizer_name,
        "history": history,
        "params": best_params,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss
    }


# ==================================================
# STEP 7: COMPARE THE THREE OPTIMIZERS
# ==================================================

models = [
    train_model("SGD"),
    train_model("Momentum"),
    train_model("Adam")
]


# ==================================================
# STEP 8: FINAL TEST EVALUATION
# ==================================================

print("\n--- Test Accuracy ---")

for model in models:
    probabilities, _ = forward(X_test, model["params"])
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, predictions)

    print(f"{model['name']}: {accuracy:.4f}")


# ==================================================
# STEP 9: PLOT TRAINING LOSS
# ==================================================

plt.figure(figsize=(9, 5))

for model in models:
    plt.plot(
        model["history"]["train_loss"],
        label=model["name"]
    )

plt.xlabel("Epoch")
plt.ylabel("Training cross-entropy loss")
plt.title("Optimizer Comparison: Training Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("19_training_loss.png", dpi=200)
plt.show()


# ==================================================
# STEP 10: PLOT VALIDATION LOSS
# ==================================================

plt.figure(figsize=(9, 5))

for model in models:
    plt.plot(
        model["history"]["val_loss"],
        label=model["name"]
    )

plt.xlabel("Epoch")
plt.ylabel("Validation cross-entropy loss")
plt.title("Optimizer Comparison: Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("19_validation_loss.png", dpi=200)
plt.show()
