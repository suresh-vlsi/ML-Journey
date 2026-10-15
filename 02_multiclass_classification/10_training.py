
# 10_training.py
# Train a multiclass classifier from scratch using NumPy.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --------------------------------------------------
# STEP 1: GENERATE THE THREE-CLASS DATASET
# --------------------------------------------------

class_0 = np.random.randn(100, 2) + [0, 0]
class_1 = np.random.randn(100, 2) + [4, 0]
class_2 = np.random.randn(100, 2) + [2, 4]

X = np.vstack((class_0, class_1, class_2))

y = np.array(
    [0] * 100 +
    [1] * 100 +
    [2] * 100
)

# Shuffle the samples before splitting
indices = np.random.permutation(len(y))
X = X[indices]
y = y[indices]

# 80% training data, 20% test data
split = int(0.8 * len(y))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Training samples:", len(y_train))
print("Test samples:", len(y_test))

# --------------------------------------------------
# STEP 2: INITIALIZE MODEL PARAMETERS
# --------------------------------------------------

num_features = 2
num_classes = 3

W = np.random.randn(num_features, num_classes) * 0.01
b = np.zeros(num_classes)

learning_rate = 0.1
epochs = 500

loss_history = []

# --------------------------------------------------
# STEP 3: DEFINE SOFTMAX
# --------------------------------------------------

def softmax(logits):
    shifted = logits - np.max(
        logits, axis=1, keepdims=True
    )

    exp_values = np.exp(shifted)

    return exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )

# --------------------------------------------------
# STEP 4: TRAIN THE MODEL
# --------------------------------------------------

N = len(y_train)
rows = np.arange(N)

for epoch in range(epochs):

    # Forward pass: calculate logits and probabilities
    logits = X_train @ W + b
    probabilities = softmax(logits)

    # Calculate average cross-entropy loss
    correct_probs = probabilities[rows, y_train]
    loss = -np.mean(np.log(np.clip(
        correct_probs, 1e-15, 1.0
    )))

    loss_history.append(loss)

    # Gradient of loss with respect to logits
    dZ = probabilities.copy()
    dZ[rows, y_train] -= 1
    dZ /= N

    # Gradients for weights and biases
    dW = X_train.T @ dZ
    db = np.sum(dZ, axis=0)

    # Gradient descent update
    W -= learning_rate * dW
    b -= learning_rate * db

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {loss:.4f}"
        )

# --------------------------------------------------
# STEP 5: EVALUATE ON UNSEEN TEST DATA
# --------------------------------------------------

test_logits = X_test @ W + b
test_probabilities = softmax(test_logits)

test_predictions = np.argmax(
    test_probabilities, axis=1
)

test_accuracy = np.mean(
    test_predictions == y_test
)

print("\nFinal training loss:", round(loss_history[-1], 4))
print("Test accuracy:", round(test_accuracy * 100, 2), "%")

# --------------------------------------------------
# STEP 6: PLOT THE LEARNING CURVE
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Multiclass Classifier Training")
plt.grid(True)
plt.tight_layout()
plt.savefig("10_training_loss.png", dpi=150)
plt.show()

print("Saved plot as 10_training_loss.png")
