
import numpy as np

# Input sequence
sequence = [1.0, 2.0, 3.0]

# RNN parameters
W_xh = 0.5
W_hh = 0.2
b_h = 0.0

# Initial hidden state
h = 0.0

# Process each input in order
for t, x in enumerate(sequence, start=1):

    # Calculate the new hidden state
    z = W_xh * x + W_hh * h + b_h
    h = np.tanh(z)

    print(f"Time step {t}")
    print(f"Input: {x:.1f}")
    print(f"Weighted sum: {z:.4f}")
    print(f"Hidden state: {h:.4f}")
    print("-" * 25)
