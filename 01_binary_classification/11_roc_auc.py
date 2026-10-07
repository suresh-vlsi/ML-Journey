import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. ACTUAL LABELS
# ============================================================

y_true = np.array([
    0, 0, 0, 0,
    1, 1, 1, 1
])


# ============================================================
# 2. MODEL PROBABILITIES
# ============================================================

y_prob = np.array([
    0.05,
    0.15,
    0.30,
    0.40,
    0.60,
    0.75,
    0.90,
    0.95
])


# ============================================================
# 3. THRESHOLDS
# ============================================================

thresholds = np.linspace(1, 0, 101)


tpr_values = []
fpr_values = []


# ============================================================
# 4. CALCULATE TPR AND FPR
#    FOR EVERY THRESHOLD
# ============================================================

for threshold in thresholds:

    # Convert probabilities into predictions
    y_pred = (y_prob >= threshold).astype(int)


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    TP = np.sum(
        (y_true == 1) &
        (y_pred == 1)
    )

    TN = np.sum(
        (y_true == 0) &
        (y_pred == 0)
    )

    FP = np.sum(
        (y_true == 0) &
        (y_pred == 1)
    )

    FN = np.sum(
        (y_true == 1) &
        (y_pred == 0)
    )


    # --------------------------------------------------------
    # TPR
    # --------------------------------------------------------

    if TP + FN > 0:
        TPR = TP / (TP + FN)
    else:
        TPR = 0


    # --------------------------------------------------------
    # FPR
    # --------------------------------------------------------

    if FP + TN > 0:
        FPR = FP / (FP + TN)
    else:
        FPR = 0


    tpr_values.append(TPR)

    fpr_values.append(FPR)


# Convert lists to NumPy arrays

tpr_values = np.array(tpr_values)

fpr_values = np.array(fpr_values)


# ============================================================
# 5. CALCULATE AUC
# ============================================================

auc = np.trapezoid(
    tpr_values,
    fpr_values
)


print("AUC =", auc)


# ============================================================
# 6. PLOT ROC CURVE
# ============================================================

plt.figure(figsize=(8, 6))


plt.plot(
    fpr_values,
    tpr_values,
    label=f"ROC Curve (AUC = {auc:.3f})"
)


# Random classifier
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


plt.xlabel("False Positive Rate (FPR)")

plt.ylabel("True Positive Rate (TPR)")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.show()