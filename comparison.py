import statistics
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl

from mandelbrot import (
    compute_mandelbrot_grid,
    compute_mandelbrot_numba,
    compute_mandelbrot_vectorized,
)
from mandelbrot_dask import mandelbrot_dask
from mandelbrot_parallel import _worker, mandelbrot_parallel


KERNEL_GPU = """
__kernel void mandelbrot_f32(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (col >= N || row >= N) return;

    float c_real = x_min + col * (x_max - x_min) / (float)(N - 1);
    float c_imag = y_min + row * (y_max - y_min) / (float)(N - 1);

    float zr = 0.0f;
    float zi = 0.0f;
    int count = 0;

    while (count < max_iter && zr*zr + zi*zi <= 4.0f) {
        float tmp = zr*zr - zi*zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;
        count++;
    }

    result[row * N + col] = count;
}
"""


def benchmark(func, *args, n_runs=3):
    times = []
    result = None

    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)

    return statistics.median(times), result


def mandelbrot_gpu_f32(N, x_min, x_max, y_min, y_max, max_iter):
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    prog = cl.Program(ctx, KERNEL_GPU).build()

    result = np.zeros((N, N), dtype=np.int32)
    result_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, result.nbytes)

    # Warm-up
    prog.mandelbrot_f32(
        queue, (64, 64), None,
        result_dev,
        np.float32(x_min), np.float32(x_max),
        np.float32(y_min), np.float32(y_max),
        np.int32(64), np.int32(max_iter),
    )
    queue.finish()

    prog.mandelbrot_f32(
        queue, (N, N), None,
        result_dev,
        np.float32(x_min), np.float32(x_max),
        np.float32(y_min), np.float32(y_max),
        np.int32(N), np.int32(max_iter),
    )
    queue.finish()

    cl.enqueue_copy(queue, result, result_dev)
    queue.finish()

    return result


if __name__ == "__main__":
    N = 4096
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.25, 1.25
    max_iter = 100

    print("=== Mandelbrot Performance Comparison ===\n")

    # Naive
    t_naive, result_naive = benchmark(
        compute_mandelbrot_grid,
        x_min, x_max, y_min, y_max,
        N, N, max_iter,
    )

    # NumPy
    t_numpy, result_numpy = benchmark(
        compute_mandelbrot_vectorized,
        x_min, x_max, y_min, y_max,
        N, max_iter,
    )

    # Numba
    _ = compute_mandelbrot_numba(x_min, x_max, y_min, y_max, 64, 64, max_iter)
    t_numba, result_numba = benchmark(
        compute_mandelbrot_numba,
        x_min, x_max, y_min, y_max,
        N, N, max_iter,
    )

    # Multiprocessing
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]

    with Pool(processes=8) as pool:
        pool.map(_worker, tiny)

        times = []
        result_parallel = None

        for _ in range(3):
            t0 = time.perf_counter()
            result_parallel = mandelbrot_parallel(
                N, x_min, x_max, y_min, y_max, max_iter, n_workers=8, n_chunks=32, pool=pool,
            )
            times.append(time.perf_counter() - t0)

    t_parallel = statistics.median(times)

    # Dask local
    times = []
    result_dask = None

    for _ in range(3):
        t0 = time.perf_counter()
        result_dask = mandelbrot_dask(
            N, x_min, x_max, y_min, y_max, max_iter=max_iter, n_chunks=32,
        )
        times.append(time.perf_counter() - t0)

    t_dask = statistics.median(times)

    # GPU OpenCL
    t_gpu, result_gpu = benchmark(
        mandelbrot_gpu_f32,
        N, x_min, x_max, y_min, y_max, max_iter,
    )

    print("Implementation        Time (s)   Speedup")
    print("------------------------------------------")
    print(f"Naive Python          {t_naive:.4f}     1.00x")
    print(f"NumPy vectorized      {t_numpy:.4f}     {t_naive / t_numpy:.2f}x")
    print(f"Numba (@njit)         {t_numba:.4f}     {t_naive / t_numba:.2f}x")
    print(f"Multiprocessing       {t_parallel:.4f}     {t_naive / t_parallel:.2f}x")
    print(f"Dask local            {t_dask:.4f}     {t_naive / t_dask:.2f}x")
    print(f"GPU OpenCL f32        {t_gpu:.4f}     {t_naive / t_gpu:.2f}x")

    print("\nParallel vs Numba:")
    print(f"Multiprocessing vs Numba: {t_numba / t_parallel:.2f}x")
    print(f"Dask vs Numba:            {t_numba / t_dask:.2f}x")
    print(f"GPU vs Numba:             {t_numba / t_gpu:.2f}x")

    print("\nValidation checks:")
    print("NumPy matches naive:", np.allclose(result_naive, result_numpy))
    print("Numba matches naive:", np.allclose(result_naive, result_numba))
    print("Parallel matches naive:", np.allclose(result_naive, result_parallel))
    print("Dask matches naive:", np.allclose(result_naive, result_dask))
    gpu_diff = np.abs(result_naive - result_gpu)
    print("GPU max difference:", gpu_diff.max())
    print("GPU different pixels:", (gpu_diff > 0).sum())
    print("GPU matches naive:", np.array_equal(result_naive, result_gpu))

    names = [
        "Naive", "NumPy", "Numba", "Multiprocessing", "Dask local", "GPU f32",
    ]

    times = [ t_naive, t_numpy, t_numba, t_parallel, t_dask, t_gpu,
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(names, times)
    plt.yscale("log")
    plt.ylabel("Runtime (s, log scale)")
    plt.title(f"MP3 Benchmark Comparison ({N}x{N}, max_iter={max_iter})")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("benchmark_mp3.png", dpi=150)
    plt.close()