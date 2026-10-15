
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==================================================
# STEP 1: CREATE THE DATASET
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

# Standardize input features using training data only.
mean_x = X_train.mean(axis=0)
std_x = X_train.std(axis=0)

X_train = (X_train - mean_x) / std_x
X_val = (X_val - mean_x) / std_x
X_test = (X_test - mean_x) / std_x

Y_train = np.eye(3)[y_train]
Y_val = np.eye(3)[y_val]


# ==================================================
# STEP 2: HELPER FUNCTIONS
# ==================================================

def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def cross_entropy(p, y_one_hot):
    eps = 1e-12
    correct_p = np.sum(p * y_one_hot, axis=1)
    return -np.mean(np.log(correct_p + eps))


# ==================================================
# STEP 3: BUILD AND TRAIN THE MLP
# ==================================================

def train_mlp(use_batchnorm, use_schedule, epochs=1000, seed=1):
    rng = np.random.default_rng(seed)

    input_size = 2
    hidden_size = 32
    output_size = 3
    m = len(X_train)

    W1 = rng.normal(
        0, np.sqrt(2 / input_size),
        (input_size, hidden_size)
    )
    b1 = np.zeros((1, hidden_size))

    W2 = rng.normal(
        0, np.sqrt(2 / hidden_size),
        (hidden_size, output_size)
    )
    b2 = np.zeros((1, output_size))

    # Batch normalization parameters.
    gamma = np.ones((1, hidden_size))
    beta = np.zeros((1, hidden_size))

    # Running statistics used for evaluation.
    running_mean = np.zeros((1, hidden_size))
    running_var = np.ones((1, hidden_size))
    momentum = 0.9
    eps = 1e-5

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []

    best_val_loss = float("inf")
    best_epoch = 0
    best_params = None

    for epoch in range(epochs):

        # ----- Learning-rate schedule -----
        if use_schedule:
            learning_rate = 0.05 * (0.5 ** (epoch // 300))
        else:
            learning_rate = 0.05

        learning_rates.append(learning_rate)

        # ----- Forward pass: hidden layer -----
        Z1 = X_train @ W1 + b1

        if use_batchnorm:
            batch_mean = Z1.mean(axis=0, keepdims=True)
            batch_var = Z1.var(axis=0, keepdims=True)

            Xhat = (Z1 - batch_mean) / np.sqrt(batch_var + eps)
            Z1_norm = gamma * Xhat + beta

            # Update running statistics for evaluation.
            running_mean = (
                momentum * running_mean
                + (1 - momentum) * batch_mean
            )
            running_var = (
                momentum * running_var
                + (1 - momentum) * batch_var
            )
        else:
            Xhat = None
            Z1_norm = Z1

        A1 = relu(Z1_norm)

        # ----- Forward pass: output layer -----
        Z2 = A1 @ W2 + b2
        P = softmax(Z2)

        # ----- Backpropagation: output layer -----
        dZ2 = (P - Y_train) / m

        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T
        dZ1_norm = dA1 * (Z1_norm > 0)

        # ----- Backpropagation: batch normalization -----
        if use_batchnorm:
            dgamma = np.sum(
                dZ1_norm * Xhat, axis=0, keepdims=True
            )
            dbeta = np.sum(
                dZ1_norm, axis=0, keepdims=True
            )

            dXhat = dZ1_norm * gamma
            inv_std = 1.0 / np.sqrt(batch_var + eps)

            dZ1 = (inv_std / m) * (
                m * dXhat
                - np.sum(dXhat, axis=0, keepdims=True)
                - Xhat * np.sum(
                    dXhat * Xhat, axis=0, keepdims=True
                )
            )
        else:
            dZ1 = dZ1_norm

        dW1 = X_train.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # ----- Gradient descent -----
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        if use_batchnorm:
            gamma -= learning_rate * dgamma
            beta -= learning_rate * dbeta

        # ----- Evaluation without gradient tracking -----
        def predict_probabilities(X_input):
            Z = X_input @ W1 + b1

            if use_batchnorm:
                Z = (
                    (Z - running_mean)
                    / np.sqrt(running_var + eps)
                )
                Z = gamma * Z + beta

            A = relu(Z)
            return softmax(A @ W2 + b2)

        train_p = predict_probabilities(X_train)
        val_p = predict_probabilities(X_val)

        train_loss = cross_entropy(train_p, Y_train)
        val_loss = cross_entropy(val_p, Y_val)

        train_pred = np.argmax(train_p, axis=1)
        val_pred = np.argmax(val_p, axis=1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(
            accuracy_score(y_train, train_pred)
        )
        val_accuracies.append(
            accuracy_score(y_val, val_pred)
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            best_params = {
                "W1": W1.copy(),
                "b1": b1.copy(),
                "W2": W2.copy(),
                "b2": b2.copy(),
                "gamma": gamma.copy(),
                "beta": beta.copy(),
                "running_mean": running_mean.copy(),
                "running_var": running_var.copy()
            }

    return {
        "name": (
            "BatchNorm + Schedule" if use_batchnorm and use_schedule
            else "BatchNorm" if use_batchnorm
            else "Baseline"
        ),
        "use_batchnorm": use_batchnorm,
        "history": {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_accuracy": train_accuracies,
            "val_accuracy": val_accuracies,
            "learning_rate": learning_rates
        },
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "params": best_params
    }


# ==================================================
# STEP 4: RUN THE THREE EXPERIMENTS
# ==================================================

models = [
    train_mlp(use_batchnorm=False, use_schedule=False),
    train_mlp(use_batchnorm=True, use_schedule=False),
    train_mlp(use_batchnorm=True, use_schedule=True)
]

for model in models:
    print(f"\n{model['name']}")
    print("Best epoch:", model["best_epoch"])
    print("Best validation loss:", round(model["best_val_loss"], 4))


# ==================================================
# STEP 5: TEST THE BEST CHECKPOINT OF EACH MODEL
# ==================================================

print("\n--- Test Results ---")

for model in models:
    p = model["params"]
    Z = X_test @ p["W1"] + p["b1"]

    if model["use_batchnorm"]:
        Z = (
            (Z - p["running_mean"])
            / np.sqrt(p["running_var"] + 1e-5)
        )
        Z = p["gamma"] * Z + p["beta"]

    A = relu(Z)
    probabilities = softmax(A @ p["W2"] + p["b2"])
    predictions = np.argmax(probabilities, axis=1)

    print(
        f"{model['name']}: "
        f"{accuracy_score(y_test, predictions):.4f}"
    )


# ==================================================
# STEP 6: PLOT LOSS CURVES
# ==================================================

plt.figure(figsize=(9, 5))

for model in models:
    plt.plot(
        model["history"]["val_loss"],
        label=model["name"]
    )

plt.xlabel("Epoch")
plt.ylabel("Validation cross-entropy loss")
plt.title("Batch Normalization and Learning-Rate Scheduling")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("18_validation_loss.png", dpi=200)
plt.show()


# ==================================================
# STEP 7: PLOT THE LEARNING-RATE SCHEDULE
# ==================================================

plt.figure(figsize=(8, 4))

plt.plot(
    models[2]["history"]["learning_rate"]
)

plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.title("Step-Decay Learning-Rate Schedule")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("18_learning_rate.png", dpi=200)
plt.show()
