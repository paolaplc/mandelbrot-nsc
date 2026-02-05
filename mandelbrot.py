"""
Mandelbrot Set Generator
Author: Paola Polanco
Course: Numerical Scientific Computing 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def mandelbrot_point(c, max_iter=100):
    z = 0
    for n in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return n
    return max_iter


def compute_mandelbrot_grid(xmin, xmax, ymin, ymax, width, height, max_iter):

    # defining the region 
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    # iterations counts 2D array 
    counts = np.zeros((height, width), dtype=int)

    # loop over points 
    for j in range(height):
        for i in range(width):
            c = complex(x[i], y[j])
            counts[j, i] = mandelbrot_point(c, max_iter)

    return counts




if __name__ == "__main__":
    print(mandelbrot_point(0))

    start = time.time()
    result = compute_mandelbrot_grid(-2, 1, -1.5, 1.5, 1024, 1024, 100)
    elapsed = time.time() - start

    print(result.shape)
    print(f"Computation took {elapsed:.3f} seconds")

    #plot
    plt.imshow(result, cmap="hot", origin="lower")
    plt.colorbar(label="Iteration count")
    plt.title("Mandelbrot Set (naive)")
    plt.savefig("mandelbrot_naive.png")
    plt.show() 







