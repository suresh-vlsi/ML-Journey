
# 13_sklearn_multiclass.py
# Multiclass classification using scikit-learn.

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# STEP 1: GENERATE THE DATASET
# --------------------------------------------------

np.random.seed(42)

class_0 = np.random.randn(100, 2) + [0, 0]
class_1 = np.random.randn(100, 2) + [4, 0]
class_2 = np.random.randn(100, 2) + [2, 4]

X = np.vstack((class_0, class_1, class_2))
y = np.array([0] * 100 + [1] * 100 + [2] * 100)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# --------------------------------------------------
# STEP 2: SPLIT INTO TRAINING AND TEST DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(y_train))
print("Test samples:", len(y_test))

# --------------------------------------------------
# STEP 3: CREATE THE MODEL
# --------------------------------------------------

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# --------------------------------------------------
# STEP 4: TRAIN THE MODEL
# --------------------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed.")

# --------------------------------------------------
# STEP 5: MAKE PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test)

# Probabilities for the first five test samples
probabilities = model.predict_proba(X_test[:5])

print("\nFirst five true labels:", y_test[:5])
print("First five predictions:", y_pred[:5])

print("\nClass order:", model.classes_)
print("Probabilities for first five samples:")
print(np.round(probabilities, 3))

# --------------------------------------------------
# STEP 6: EVALUATE THE MODEL
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2]
)

print("\nTest accuracy:", round(accuracy, 4))

print("\nConfusion matrix:")
print(cm)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0
    )
)

# --------------------------------------------------
# STEP 7: PLOT THE CONFUSION MATRIX
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 5))

image = ax.imshow(cm, interpolation="nearest")
fig.colorbar(image, ax=ax)

ax.set(
    xticks=np.arange(3),
    yticks=np.arange(3),
    xticklabels=[0, 1, 2],
    yticklabels=[0, 1, 2],
    xlabel="Predicted class",
    ylabel="True class",
    title="Multiclass Confusion Matrix"
)

for row in range(3):
    for col in range(3):
        ax.text(
            col, row, str(cm[row, col]),
            ha="center", va="center"
        )

plt.tight_layout()
plt.savefig("13_confusion_matrix.png", dpi=150)
plt.show()

print("Saved 13_confusion_matrix.png")
