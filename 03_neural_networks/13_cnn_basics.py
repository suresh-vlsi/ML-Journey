
import numpy as np
import matplotlib.pyplot as plt

# 1. A small grayscale image
image = np.array([
    [1, 2, 0, 1, 3],
    [3, 1, 2, 0, 1],
    [0, 1, 3, 2, 0],
    [2, 4, 1, 3, 1],
    [1, 0, 2, 4, 3]
], dtype=float)

# 2. A 3x3 vertical-edge filter
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=float)


# 3. Implement valid cross-correlation
def convolve_valid(image, kernel, stride=1):
    height, width = image.shape
    kh, kw = kernel.shape

    out_h = (height - kh) // stride + 1
    out_w = (width - kw) // stride + 1

    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            patch = image[
                i * stride:i * stride + kh,
                j * stride:j * stride + kw
            ]

            output[i, j] = np.sum(patch * kernel)

    return output


# 4. Convolution
feature_map = convolve_valid(image, kernel)

# 5. ReLU activation
relu_map = np.maximum(0, feature_map)


# 6. 2x2 max pooling with stride 2
def max_pool_2x2(matrix):
    height, width = matrix.shape

    out_h = height // 2
    out_w = width // 2

    pooled = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            patch = matrix[
                2*i:2*i+2,
                2*j:2*j+2
            ]

            pooled[i, j] = np.max(patch)

    return pooled


pooled_map = max_pool_2x2(relu_map)

# 7. Display numerical results
print("INPUT IMAGE:")
print(image)

print("\nKERNEL:")
print(kernel)

print("\nFEATURE MAP:")
print(feature_map)

print("\nAFTER RELU:")
print(relu_map)

print("\nAFTER MAX POOLING:")
print(pooled_map)

# 8. Plot all stages
fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))

stages = [
    (image, "Input Image"),
    (feature_map, "Feature Map"),
    (relu_map, "After ReLU"),
    (pooled_map, "After Max Pooling")
]

for ax, (matrix, title) in zip(axes, stages):
    ax.imshow(matrix, cmap="gray", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f"{matrix[i, j]:.0f}",
                ha="center", va="center", color="red"
            )

plt.tight_layout()
plt.show()
