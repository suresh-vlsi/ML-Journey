
# Lesson 2: Feature transformation and XOR

def step(z):
    """Return 1 if z >= 0, otherwise return 0."""
    return 1 if z >= 0 else 0


# Four XOR input examples
data = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0)
]

print("METHOD A: TRANSFORMED FEATURES")
print("x1 x2 x1*x2 z prediction")

for x1, x2, target in data:

    # Create a new feature
    interaction = x1 * x2

    # Linear classifier in transformed feature space
    z = x1 + x2 - 2 * interaction - 0.5

    # Threshold activation
    prediction = step(z)

    print(x1, x2, interaction, z, prediction)


print("\nMETHOD B: TWO HIDDEN NEURONS")
print("x1 x2 h1 h2 prediction")

for x1, x2, target in data:

    # Hidden-layer calculations
    h1 = step(x1 - x2 - 0.5)
    h2 = step(-x1 + x2 - 0.5)

    # Output-layer calculation
    prediction = step(h1 + h2 - 0.5)

    print(x1, x2, h1, h2, prediction)
