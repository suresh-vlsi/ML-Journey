
import math

# Inputs
x1 = 3.0
x2 = 2.0

# Weights and bias
w1 = 0.8
w2 = 0.5
b = -3.0

# Step 1: Calculate the weighted sum
z = w1 * x1 + w2 * x2 + b

# Step 2: Apply sigmoid activation
output = 1 / (1 + math.exp(-z))

# Step 3: Convert the output to a binary decision
prediction = "PASS" if output >= 0.5 else "FAIL"

print("Weighted sum:", z)
print("Neuron output:", output)
print("Prediction:", prediction)
