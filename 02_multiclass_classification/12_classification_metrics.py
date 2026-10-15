
# 12_classification_metrics.py
# Multiclass evaluation from scratch using NumPy.

import numpy as np

# True labels: 10 samples from each class
y_true = np.array(
    [0] * 10 + [1] * 10 + [2] * 10
)

# Predictions constructed to match our example confusion matrix:
# Actual Class 0: 8 predicted 0, 1 predicted 1, 1 predicted 2
# Actual Class 1: 2 predicted 0, 6 predicted 1, 2 predicted 2
# Actual Class 2: 1 predicted 0, 1 predicted 1, 8 predicted 2

y_pred = np.array(
    [0] * 8 + [1] + [2] +
    [0] * 2 + [1] * 6 + [2] * 2 +
    [0] + [1] + [2] * 8
)

num_classes = 3

# STEP 1: Build confusion matrix
cm = np.zeros((num_classes, num_classes), dtype=int)

for actual, predicted in zip(y_true, y_pred):
    cm[actual, predicted] += 1

print("Confusion matrix:")
print(cm)

# STEP 2: Calculate true positives, false positives,
# and false negatives for each class.

tp = np.diag(cm)
fp = cm.sum(axis=0) - tp
fn = cm.sum(axis=1) - tp

# STEP 3: Calculate precision and recall safely
precision = np.divide(
    tp, tp + fp,
    out=np.zeros(num_classes, dtype=float),
    where=(tp + fp) != 0
)

recall = np.divide(
    tp, tp + fn,
    out=np.zeros(num_classes, dtype=float),
    where=(tp + fn) != 0
)

# STEP 4: Calculate F1-score
f1 = np.divide(
    2 * precision * recall,
    precision + recall,
    out=np.zeros(num_classes, dtype=float),
    where=(precision + recall) != 0
)

# STEP 5: Calculate overall accuracy
accuracy = np.mean(y_true == y_pred)

print("\nPer-class metrics:")
for class_id in range(num_classes):
    print(
        f"Class {class_id}: "
        f"Precision={precision[class_id]:.4f}, "
        f"Recall={recall[class_id]:.4f}, "
        f"F1={f1[class_id]:.4f}"
    )

print("\nAccuracy:", round(accuracy, 4))
print("Macro precision:", round(np.mean(precision), 4))
print("Macro recall:", round(np.mean(recall), 4))
print("Macro F1:", round(np.mean(f1), 4))
