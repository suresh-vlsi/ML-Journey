import numpy as np

gradient = np.array([30.0, 40.0])

max_norm = 5.0
norm = np.linalg.norm(gradient)

if norm > max_norm:
    gradient = gradient * (max_norm / norm)

print("Clipped gradient:", gradient)
print("New norm:", np.linalg.norm(gradient))