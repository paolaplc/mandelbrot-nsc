import numpy as np
import matplotlib.pyplot as plt

from mandelbrot import (
    compute_mandelbrot_grid,
    compute_mandelbrot_vectorized,
    compute_mandelbrot_hybrid,
    compute_mandelbrot_numba,
    benchmark
)

try:
    profile
except NameError:
    def profile(func):
        return func


print("Numba Benchmark")

# baseline
t_naive, result_naive = benchmark(
    compute_mandelbrot_grid,
    -2, 1, -1.5, 1.5,
    1024, 1024, 100,
    n_runs=3
)

print("\nNumPy:")
t_numpy, result_numpy = benchmark(
    compute_mandelbrot_vectorized,
    -2, 1, -1.5, 1.5,
    1024, 100,
    n_runs=3
)
print("NumPy matches naive:", np.allclose(result_naive, result_numpy))

# warm-up (JIT compile) -- DO NOT time this
_ = compute_mandelbrot_hybrid(-2, 1, -1.5, 1.5, 64, 64, 100)
_ = compute_mandelbrot_numba(-2, 1, -1.5, 1.5, 64, 64, 100)

print("\nNumba Hybrid:")
t_hybrid, result_hybrid = benchmark(
    compute_mandelbrot_hybrid,
    -2, 1, -1.5, 1.5,
    1024, 1024, 100,
    n_runs=3
)

print("\nNumba Fully compiled:")
t_numba, result_numba = benchmark(
    compute_mandelbrot_numba,
    -2, 1, -1.5, 1.5,
    1024, 1024, 100,
    n_runs=3
)

print("\nValidation vs naive:")
print("Hybrid matches naive:", np.allclose(result_naive, result_hybrid))
print("Numba matches naive:", np.allclose(result_naive, result_numba))

print("\nSpeedups vs naive:")
print(f"NumPy speedup:  {t_naive/t_numpy:.2f}x")
print(f"Hybrid speedup: {t_naive/t_hybrid:.2f}x")
print(f"Numba speedup:  {t_naive/t_numba:.2f}x")


# plot
plt.figure()
plt.imshow(result_numba, cmap="hot", origin="lower")
plt.colorbar(label="Iteration count")
plt.title(f"Mandelbrot (Numba) 1024x1024  time={t_numba:.3f}s")
plt.savefig("mandelbrot_numba.png", dpi=150)
plt.show()