
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate input values
x = np.linspace(-1, 3, 200)

# Step 2: Network A - two linear layers
hidden_linear = 2 * x + 1
output_linear = 3 * hidden_linear - 2

# Step 3: Network B - a ReLU hidden layer
hidden_relu = np.maximum(0, 2 * x - 2)
output_relu = 3 * hidden_relu + 1

# Step 4: Print a few numerical comparisons
print("x | Linear output | ReLU output")

for value in [0, 0.5, 1, 1.5, 2, 3]:
    linear_y = 6 * value + 1
    relu_y = 3 * max(0, 2 * value - 2) + 1

    print(f"{value:3.1f} | {linear_y:13.2f} | {relu_y:11.2f}")

# Step 5: Plot the linear network
plt.figure(figsize=(8, 5))
plt.plot(x, output_linear, label="Two linear layers")
plt.xlabel("Input x")
plt.ylabel("Output y")
plt.title("Linear Network")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Step 6: Plot the nonlinear network
plt.figure(figsize=(8, 5))
plt.plot(x, output_relu, label="ReLU hidden layer")
plt.xlabel("Input x")
plt.ylabel("Output y")
plt.title("Nonlinear Network")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
