
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Generate a nonlinear binary-classification dataset
X, y = make_moons(
    n_samples=800,
    noise=0.30,
    random_state=42
)

# 2. Split off the test set first
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Create training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.25,
    random_state=42,
    stratify=y_trainval
)

# 4. Train models with different L2 strengths
regularization_values = [1e-5, 1e-2]

models = {}

for alpha in regularization_values:
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(100, 100),
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=0.005,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42
        )
    )

    model.fit(X_train, y_train)
    models[alpha] = model

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    print(f"\nL2 regularization alpha = {alpha}")
    print(f"Training accuracy:   {accuracy_score(y_train, train_pred):.3f}")
    print(f"Validation accuracy: {accuracy_score(y_val, val_pred):.3f}")
    print(f"Test accuracy:      {accuracy_score(y_test, test_pred):.3f}")

# 5. Plot the training loss curves
plt.figure(figsize=(9, 5))

for alpha, model in models.items():
    mlp = model.named_steps["mlpclassifier"]
    plt.plot(
        mlp.loss_curve_,
        label=f"L2 alpha = {alpha}"
    )

plt.xlabel("Training iteration")
plt.ylabel("Training loss")
plt.title("MLP Training Loss with Different L2 Strengths")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
