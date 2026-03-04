import numpy as np
import matplotlib.pyplot as plt

#Benchmark, speed ups and validation

from mandelbrot import (
    compute_mandelbrot_grid,
    compute_mandelbrot_vectorized,
    compute_mandelbrot_numba,
    compute_mandelbrot_numba_f32,
    compute_mandelbrot_numba_f64,
    benchmark
)


print("Benchmark: naive vs NumPy vs Numba")

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

# warm-up 
_ = compute_mandelbrot_numba(-2, 1, -1.5, 1.5, 64, 64, 100)

print("\nNumba (naive + @njit):")
t_numba, result_numba = benchmark(
    compute_mandelbrot_numba,
    -2, 1, -1.5, 1.5,
    1024, 1024, 100,
    n_runs=3
)

print("\nValidation vs naive:")
print("Numba matches naive:", np.allclose(result_naive, result_numba))

print("\nSpeedups vs naive:")
print(f"NumPy speedup:  {t_naive/t_numpy:.2f}x")
print(f"Numba speedup:  {t_naive/t_numba:.2f}x")

#plot
plt.figure()
plt.imshow(result_numba, cmap="hot", origin="lower")
plt.colorbar(label="Iteration count")
plt.title(f"Mandelbrot (Numba) 1024x1024  time={t_numba:.3f}s")
plt.savefig("mandelbrot_numba.png", dpi=150)
plt.show()


#Milestone 4: float32 vs float64 (Numba)

from mandelbrot import compute_mandelbrot_numba_f32, compute_mandelbrot_numba_f64

# warm-up both (JIT compile) -- DO NOT time this
_ = compute_mandelbrot_numba_f32(-2, 1, -1.5, 1.5, 64, 64, 100)
_ = compute_mandelbrot_numba_f64(-2, 1, -1.5, 1.5, 64, 64, 100)

print("\nNumba float64:")
t_f64, r64 = benchmark(
    compute_mandelbrot_numba_f64,
    -2, 1, -1.5, 1.5,
    1024, 1024, 100,
    n_runs=3
)

print("\nNumba float32:")
t_f32, r32 = benchmark(
    compute_mandelbrot_numba_f32,
    -2, 1, -1.5, 1.5,
    1024, 1024, 100,
    n_runs=3
)

print(f"\nSpeedup float32 vs float64: {t_f64/t_f32:.2f}x")
diff = np.abs(r32 - r64)
print("Max diff float32 vs float64:", diff.max())
print("Different pixels:", (diff > 0).sum())

#pltos
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(r64, cmap="hot", origin="lower")
axes[0].set_title("Numba float64")
axes[0].axis("off")

axes[1].imshow(r32, cmap="hot", origin="lower")
axes[1].set_title("Numba float32")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("precision_comparison_f32_f64.png", dpi=150)
plt.show()

