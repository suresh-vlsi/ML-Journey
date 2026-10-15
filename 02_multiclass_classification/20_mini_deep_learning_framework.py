
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# PART 1: DATASET
# =========================================================

def make_three_spirals(samples_per_class=300, noise=0.08, seed=42):
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []

    for class_id in range(3):
        t = np.linspace(0, 1, samples_per_class)
        radius = 0.2 + 1.8 * t
        angle = 2 * np.pi * 1.25 * t + 2 * np.pi * class_id / 3

        points = np.column_stack([
            radius * np.cos(angle),
            radius * np.sin(angle)
        ])

        points += rng.normal(0, noise, points.shape)

        X_parts.append(points)
        y_parts.append(np.full(samples_per_class, class_id))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    order = rng.permutation(len(y))
    return X[order], y[order]


X, y = make_three_spirals()

# 64% train, 16% validation, 20% test.
X_train_all, X_test, y_train_all, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_all,
    y_train_all,
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


# =========================================================
# PART 2: LAYERS
# =========================================================

class Dense:
    """Fully connected layer: output = input @ W + b."""

    def __init__(self, input_size, output_size, seed=1):
        rng = np.random.default_rng(seed)

        # He-style initialization for a layer followed by ReLU.
        self.W = rng.normal(
            0,
            np.sqrt(2 / input_size),
            size=(input_size, output_size)
        )
        self.b = np.zeros((1, output_size))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.input = None

    def forward(self, X):
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad_output):
        # grad_output is dLoss/dOutput.
        n = self.input.shape[0]

        self.dW = self.input.T @ grad_output
        self.db = np.sum(grad_output, axis=0, keepdims=True)

        # Gradient passed to the previous layer.
        grad_input = grad_output @ self.W.T

        return grad_input

    def parameters(self):
        return [
            (self.W, self.dW),
            (self.b, self.db)
        ]


class ReLU:
    """Rectified Linear Unit activation."""

    def __init__(self):
        self.input = None

    def forward(self, X):
        self.input = X
        return np.maximum(0, X)

    def backward(self, grad_output):
        return grad_output * (self.input > 0)


# =========================================================
# PART 3: NEURAL NETWORK
# =========================================================

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.layers = [
            Dense(input_size, hidden_size, seed=1),
            ReLU(),
            Dense(hidden_size, output_size, seed=2)
        ]

    def forward(self, X):
        output = X

        for layer in self.layers:
            output = layer.forward(output)

        # Return logits, not probabilities.
        return output

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def parameters(self):
        result = []

        for layer in self.layers:
            if isinstance(layer, Dense):
                result.extend(layer.parameters())

        return result


# =========================================================
# PART 4: SOFTMAX AND CROSS-ENTROPY
# =========================================================

def softmax(logits):
    # Subtract maximum to improve numerical stability.
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)

    return exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )


def cross_entropy_and_gradient(logits, labels):
    n = len(labels)
    probabilities = softmax(logits)

    # Mean cross-entropy.
    correct_probabilities = probabilities[
        np.arange(n), labels
    ]

    loss = -np.mean(
        np.log(np.clip(correct_probabilities, 1e-12, 1.0))
    )

    # Gradient of mean softmax cross-entropy with respect to logits.
    grad = probabilities.copy()
    grad[np.arange(n), labels] -= 1
    grad /= n

    return loss, grad, probabilities


# =========================================================
# PART 5: REUSABLE OPTIMIZER
# =========================================================

class Optimizer:
    def __init__(
        self,
        name="adam",
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        momentum=0.9
    ):
        self.name = name.lower()
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentum = momentum

        self.step = 0
        self.state = {}

        if self.name not in ("sgd", "momentum", "adam"):
            raise ValueError("Choose sgd, momentum, or adam.")

    def update(self, parameters):
        self.step += 1

        for index, (param, grad) in enumerate(parameters):
            if self.name == "sgd":
                param -= self.lr * grad

            elif self.name == "momentum":
                if index not in self.state:
                    self.state[index] = np.zeros_like(param)

                velocity = self.state[index]
                velocity *= self.momentum
                velocity += grad

                param -= self.lr * velocity

            elif self.name == "adam":
                if index not in self.state:
                    self.state[index] = {
                        "m": np.zeros_like(param),
                        "v": np.zeros_like(param)
                    }

                state = self.state[index]
                state["m"] = (
                    self.beta1 * state["m"]
                    + (1 - self.beta1) * grad
                )
                state["v"] = (
                    self.beta2 * state["v"]
                    + (1 - self.beta2) * grad**2
                )

                m_hat = state["m"] / (
                    1 - self.beta1**self.step
                )
                v_hat = state["v"] / (
                    1 - self.beta2**self.step
                )

                param -= (
                    self.lr * m_hat
                    / (np.sqrt(v_hat) + self.epsilon)
                )


# =========================================================
# PART 6: TRAINING AND EVALUATION FUNCTIONS
# =========================================================

def predict(network, X):
    logits = network.forward(X)
    return np.argmax(logits, axis=1)


def evaluate(network, X, y):
    logits = network.forward(X)
    loss, _, probabilities = cross_entropy_and_gradient(
        logits, y
    )

    predictions = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y, predictions)

    return loss, accuracy


def train(
    network,
    optimizer,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=1500
):
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

        # Forward pass.
        logits = network.forward(X_train)

        # Loss and output gradient.
        loss, grad, _ = cross_entropy_and_gradient(
            logits, y_train
        )

        # Backpropagate through all layers.
        network.backward(grad)

        # Update weights and biases.
        optimizer.update(network.parameters())

        # Evaluate after updating the parameters.
        train_loss, train_accuracy = evaluate(
            network, X_train, y_train
        )
        val_loss, val_accuracy = evaluate(
            network, X_val, y_val
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        # Save the best validation-loss checkpoint.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            best_parameters = [
                (layer.W.copy(), layer.b.copy())
                for layer in network.layers
                if isinstance(layer, Dense)
            ]

        if (epoch + 1) % 250 == 0:
            print(
                f"Epoch {epoch + 1:4d} | "
                f"Train loss: {train_loss:.4f} | "
                f"Validation loss: {val_loss:.4f} | "
                f"Validation accuracy: {val_accuracy:.4f}"
            )

    # Restore the best checkpoint.
    dense_layers = [
        layer for layer in network.layers
        if isinstance(layer, Dense)
    ]

    for layer, (W, b) in zip(dense_layers, best_parameters):
        layer.W = W
        layer.b = b

    print("\nBest validation epoch:", best_epoch)
    print("Best validation loss:", round(best_val_loss, 4))

    return history


# =========================================================
# PART 7: BUILD AND TRAIN THE FINAL NETWORK
# =========================================================

network = NeuralNetwork(
    input_size=2,
    hidden_size=32,
    output_size=3
)

optimizer = Optimizer(
    name="adam",
    learning_rate=0.01
)

history = train(
    network,
    optimizer,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=1500
)


# =========================================================
# PART 8: FINAL TEST EVALUATION
# =========================================================

test_loss, test_accuracy = evaluate(
    network, X_test, y_test
)

print("\n========== FINAL TEST RESULTS ==========")
print("Test loss:", round(test_loss, 4))
print("Test accuracy:", round(test_accuracy, 4))

test_predictions = predict(network, X_test)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)


# =========================================================
# PART 9: PLOT TRAINING AND VALIDATION LOSS
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(history["train_loss"], label="Training loss")
plt.plot(history["val_loss"], label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("Mini Deep-Learning Framework")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("20_training_loss.png", dpi=200)
plt.show()


# =========================================================
# PART 10: VISUALIZE DECISION BOUNDARIES
# =========================================================

x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 250),
    np.linspace(y_min, y_max, 250)
)

grid = np.column_stack((xx.ravel(), yy.ravel()))
grid_scaled = (grid - mean) / std

grid_predictions = predict(network, grid_scaled)
regions = grid_predictions.reshape(xx.shape)

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

plt.title("Final MLP Decision Boundaries")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("20_decision_boundaries.png", dpi=200)
plt.show()
