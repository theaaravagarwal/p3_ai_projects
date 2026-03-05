import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

matplotlib.use("Agg")

# Image of an "X"
image = np.array(
    [
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1],
    ],
    dtype=np.uint8,
)

kernel = np.array(
    [
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ]
)

# Convolution is the core operation used in CNN filters.
result = convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0)
print("Result of convolution:")
print(result)

plt.figure(figsize=(4, 4))
plt.title("Input Image")
plt.imshow(image, cmap="gray", vmin=image.min(), vmax=image.max())
plt.axis("off")
plt.tight_layout()
plt.savefig("input_image.png", dpi=150)
plt.close()

plt.figure(figsize=(4, 4))
plt.title("Convolution Output")
plt.imshow(result, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.savefig("convolution_result.png", dpi=150)
plt.close()
