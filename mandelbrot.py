"""
Mandelbrot Set Generator
Author: Paola Polanco
Course: Numerical Scientific Computing 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import statistics


#lesson2 milestone 1
def create_complex_grid(xmin, xmax, ymin, ymax, n):
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymin, ymax, n)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    return C



#lesson2 benchmark 
def benchmark(func, *args, n_runs=3):
    """Time func, return median of n_runs."""
    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)

    median_t = statistics.median(times)
    print(
        f"Median: {median_t:.4f}s "
        f"(min={min(times):.4f}, max={max(times):.4f})"
    )
    return median_t, result


# -----------------------------------

#Naive approach  lesson 1 
def mandelbrot_point(c, max_iter=100):
    z = 0j
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

# ---------------------------------
#Lesson2 milestone 2: vectorize 
def compute_mandelbrot_vectorized(xmin, xmax, ymin, ymax, n, max_iter):
    C = create_complex_grid(xmin, xmax, ymin, ymax, n)

    Z = np.zeros_like(C)              
    M = np.zeros(C.shape, dtype=int)  

    for _ in range(max_iter):        
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1

    return M



# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    #lesson2 milestone 1
    C = create_complex_grid(-2, 1, -1.5, 1.5, 1024)
    print(f"Shape: {C.shape}")
    print(f"Type:  {C.dtype}")

    # sanity check
    print("mandelbrot_point(0) =", mandelbrot_point(0))

    # benchmark naive baseline (>=3 runs, perf_counter)
    t_med, result = benchmark(
        compute_mandelbrot_grid,
        -2, 1, -1.5, 1.5,
        1024, 1024, 100,
        n_runs=3
    )

    print("Result shape:", result.shape)

    print("\nVectorized:")
    t_vec, result_vec = benchmark(compute_mandelbrot_vectorized, -2, 1, -1.5, 1.5, 1024, 100, n_runs=3)
    print(result_vec.shape)

  # Validate (slide version)
    if np.allclose(result, result_vec):
        print("Results match!")
    else:
        print("Results differ!")

    diff = np.abs(result - result_vec)
    print(f"Max difference: {diff.max()}")
    print(f"Different pixels: {(diff > 0).sum()}")

    print(f"Speedup: {t_med / t_vec:.2f}x")

    # plot 
    plt.imshow(result_vec, cmap="hot", origin="lower")
    plt.colorbar(label="Iteration count")
    plt.title(f"Mandelbrot Set (vectorized) median={t_vec:.3f}s")
    plt.savefig("mandelbrot_vectorized.png", dpi=150)
    plt.show()





