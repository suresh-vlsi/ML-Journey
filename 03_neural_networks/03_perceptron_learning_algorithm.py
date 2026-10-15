
# Lesson 3: Perceptron learning algorithm
# Example: Learn the AND logic gate

# Step 1: Create the training dataset
data = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1)
]

# Step 2: Initialize parameters
w1 = 0.0
w2 = 0.0
b = 0.0

# Step 3: Set learning parameters
learning_rate = 1.0
max_epochs = 20


# Step 4: Define the activation function
def step(z):
    return 1 if z >= 0 else 0


# Step 5: Train the perceptron
for epoch in range(max_epochs):

    mistakes = 0

    for x1, x2, target in data:

        # Forward propagation
        z = w1 * x1 + w2 * x2 + b
        prediction = step(z)

        # Calculate prediction error
        error = target - prediction

        # Update parameters if prediction is wrong
        w1 = w1 + learning_rate * error * x1
        w2 = w2 + learning_rate * error * x2
        b = b + learning_rate * error

        if error != 0:
            mistakes += 1

    print(
        f"Epoch {epoch + 1}: "
        f"mistakes={mistakes}, "
        f"w1={w1:.1f}, w2={w2:.1f}, b={b:.1f}"
    )

    # Stop when every training example is classified correctly
    if mistakes == 0:
        break


# Step 6: Test the trained model
print("\nFinal predictions:")

for x1, x2, target in data:
    z = w1 * x1 + w2 * x2 + b
    prediction = step(z)

    print(
        f"Input=({x1}, {x2}), "
        f"Target={target}, Prediction={prediction}"
    )
