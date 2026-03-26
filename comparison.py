import numpy as np
import time
import statistics
from multiprocessing import Pool
from mandelbrot_parallel import _worker, mandelbrot_parallel


from mandelbrot import (
    compute_mandelbrot_grid,
    compute_mandelbrot_vectorized,
    compute_mandelbrot_numba,
)




def benchmark(func, *args, n_runs=3):
    times = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)

    return statistics.median(times), result


if __name__ == "__main__":

    N = 1024
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.25, 1.25
    max_iter = 100

    print("=== Mandelbrot Performance Comparison ===\n")

    # Naive
    t_naive, result_naive = benchmark(
        compute_mandelbrot_grid,
        x_min, x_max, y_min, y_max,
        N, N, max_iter
    )

    # NumPy
    t_numpy, result_numpy = benchmark(
        compute_mandelbrot_vectorized,
        x_min, x_max, y_min, y_max,
        N, max_iter
    )

    # Numba
    _ = compute_mandelbrot_numba(x_min, x_max, y_min, y_max, 64, 64, max_iter)
    t_numba, result_numba = benchmark(
        compute_mandelbrot_numba,
        x_min, x_max, y_min, y_max,
        N, N, max_iter
    )
 
    #Multiprocessing
    
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]

    with Pool(processes=8) as pool:
        pool.map(_worker, tiny)
    
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            result_parallel = mandelbrot_parallel(
                N, x_min, x_max, y_min, y_max,
                max_iter,
                n_workers=8,
                n_chunks=32,
                pool = pool
            )
            times.append(time.perf_counter() - t0)

    t_parallel = statistics.median(times)

    print("Implementation        Time (s)   Speedup")
    print("------------------------------------------")

    print(f"Naive Python          {t_naive:.4f}     1.00x")
    print(f"NumPy vectorized      {t_numpy:.4f}     {t_naive/t_numpy:.2f}x")
    print(f"Numba (@njit)         {t_numba:.4f}     {t_naive/t_numba:.2f}x")
    print(f"Multiprocessing       {t_parallel:.4f}     {t_naive/t_parallel:.2f}x")

    print("\nValidation checks:")
    print("NumPy matches naive:", np.allclose(result_naive, result_numpy))
    print("Numba matches naive:", np.allclose(result_naive, result_numba))
    print("Parallel matches naive:", np.allclose(result_naive, result_parallel))