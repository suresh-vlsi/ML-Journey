@"
# ML Journey

A from-scratch machine learning journey focused on understanding the mathematics, implementing the algorithms with NumPy, and experimenting with their behavior.

The goal is:

**Intuition → Mathematics → Python/NumPy → Experiment → Visualization**

---

## 01 — Binary Classification From Scratch

### Binary Classification From Scratch — ML Foundations

This stage builds binary classification from the ground up without scikit-learn.

The repository contains two complementary parts:

- **Python files** — progressive implementations of individual concepts
- **Interactive Jupyter/Colab notebook** — an experimental environment where parameters and hyperparameters can be changed interactively

### Topics Covered

- Dataset generation
- Features and labels
- Linear models
- Logistic regression
- Single neuron
- Sigmoid activation
- Binary cross-entropy
- Gradient derivation
- Gradient descent
- Mini-batch gradient descent
- Learning rate
- Epochs
- Train / validation / test split
- Decision boundaries
- Classification threshold
- Confusion matrix
- Accuracy
- Precision
- Recall
- F1 score
- TPR / FPR
- ROC curve
- AUC
- XOR problem
- Linear separability
- Hidden layers
- Forward propagation
- Backpropagation
- ReLU
- Sigmoid hidden layers
- Multilayer perceptron
- L2 regularization
- Overfitting and generalization

---

## Interactive ML Laboratory

The main notebook is designed for experimentation rather than simply reading code.

### Interactive parameters

You can change:

| Parameter | What you can experiment with |
|---|---|
| Dataset | Linear, Nonlinear Circles, XOR |
| Samples | Number of data points |
| Features | 2–8 input features |
| Noise | Dataset difficulty |
| Random seed | Dataset variation |
| Learning rate | Training speed/stability |
| Epochs | Training duration |
| Batch size | Batch vs mini-batch learning |
| Threshold | Precision/recall trade-off |
| L2 λ | Regularization strength |
| Hidden units | MLP capacity |
| Hidden activation | ReLU / Sigmoid |
| MLP learning rate | Neural-network optimization |
| MLP epochs | Neural-network training |

### Visualizations

The notebook generates:

- Dataset visualization
- Training loss curves
- Decision boundary
- Confusion matrix
- ROC curve
- AUC
- Predicted probability distributions
- MLP hidden-to-output weights

### Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suresh-vlsi/ML-Journey/blob/main/01_binary_classification/Binary_Classification_From_Scratch_ML_Journey.ipynb)

**[Open Interactive Binary Classification Lab in Google Colab](https://colab.research.google.com/github/suresh-vlsi/ML-Journey/blob/main/01_binary_classification/Binary_Classification_From_Scratch_ML_Journey.ipynb)**

---
