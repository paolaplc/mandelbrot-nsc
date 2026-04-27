"""
Mandelbrot Set Generator
Author: Paola Polanco
Course: Numerical Scientific Computing 2026
"""

import numpy as np
import time
import statistics
import cProfile
import pstats
from numba import njit

try:
    profile
except NameError:
    def profile(func):
        return func


#-------------------------
# Helping functions
#-------------------------

#lesson2 milestone 1
def create_complex_grid(xmin: float, xmax: float, ymin: float, ymax: float, n: int) -> np.ndarray:
    """Create complex grid from domain."""
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymin, ymax, n)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    return C


#lesson2 benchmark 
def benchmark(func, *args, n_runs: int = 3) -> tuple:
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


#lesson 3 milestone 1 profile fuction 
def profile_function(callable_fn, prof_filename: str, top_n: int = 10) -> None:
    """Profile function using cProfile."""

    cProfile.runctx(
        "callable_fn()",
        globals(),
        {"callable_fn": callable_fn},
        prof_filename
    )
    stats = pstats.Stats(prof_filename)
    stats.sort_stats("cumulative")
    stats.print_stats(top_n)




#-------------------------
# Naive implementation
#-------------------------

#l1: Naive approach  lesson 1 
def mandelbrot_point(c: complex, max_iter: int = 100) -> int:
    """Return escape iteration for one complex point."""
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter


@profile
def compute_mandelbrot_grid(xmin, xmax, ymin, ymax, width, height, max_iter):
    """Compute Mandelbrot set using naive loops."""

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





#-------------------------
# Numpy vectorized implementation
#-------------------------

#Lesson2 milestone 2: vectorize 
def compute_mandelbrot_vectorized(xmin: float, xmax: float, ymin: float, ymax: float, n: int, max_iter: int) -> np.ndarray:
    """Compute Mandelbrot set using NumPy."""
    C = create_complex_grid(xmin, xmax, ymin, ymax, n)

    Z = np.zeros_like(C)              
    M = np.zeros(C.shape, dtype=np.int32)  

    for _ in range(max_iter):        
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1

    return M

# ---------------------------------




#-------------------------
# Numba implementation
#-------------------------

#lesson3: milestone 3 : Numba 
#replace with milestone 4 sames as float64
# full compiled
@njit
def compute_mandelbrot_numba( xmin: float, xmax: float,  ymin: float,  ymax: float, width: int,  height: int,  max_iter: int = 100,
) -> np.ndarray:
    """Compute Mandelbrot set using Numba."""

    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    result = np.zeros((height, width), dtype=np.int32)

    for j in range(height):
        for i in range(width):

            c = x[i] + 1j*y[j]

            z = 0j
            n = 0

            while n < max_iter and (z.real*z.real + z.imag*z.imag) <= 4.0:
                z = z*z + c
                n += 1

            result[j, i] = n

    return result


#lesson3 milestone 4: floating point precision

@njit
def compute_mandelbrot_numba_f32( xmin: float, xmax: float, ymin: float, ymax: float, width: int, height: int, max_iter: int = 100,
) -> np.ndarray:
    """Compute Mandelbrot set with float32 (Numba)."""
    x = np.linspace(xmin, xmax, width).astype(np.float32)
    y = np.linspace(ymin, ymax, height).astype(np.float32)
    result = np.zeros((height, width), dtype=np.int32)

    for j in range(height):
        for i in range(width):
            cr = x[i]
            ci = y[j]
            zr = np.float32(0.0)
            zi = np.float32(0.0)
            n = 0

            while n < max_iter and (zr*zr + zi*zi) <= np.float32(4.0):
                # (zr + i zi)^2 + (cr + i ci)
                zr_new = zr*zr - zi*zi + cr
                zi = np.float32(2.0)*zr*zi + ci
                zr = zr_new
                n += 1

            result[j, i] = n
    return result


@njit
def compute_mandelbrot_numba_f64(xmin: float, xmax: float, ymin: float, ymax: float, width: int, height: int, max_iter: int = 100) -> np.ndarray:
    """Compute Mandelbrot set with float64 (Numba)."""
    x = np.linspace(xmin, xmax, width).astype(np.float64)
    y = np.linspace(ymin, ymax, height).astype(np.float64)
    result = np.zeros((height, width), dtype=np.int32)

    for j in range(height):
        for i in range(width):
            cr = x[i]
            ci = y[j]
            zr = 0.0
            zi = 0.0
            n = 0

            while n < max_iter and (zr*zr + zi*zi) <= 4.0:
                zr_new = zr*zr - zi*zi + cr
                zi = 2.0*zr*zi + ci
                zr = zr_new
                n += 1

            result[j, i] = n
    return result


# --------------------------------


