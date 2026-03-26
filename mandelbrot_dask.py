import numpy as np
from dask import delayed
from dask.distributed import Client, LocalCluster
import dask
import time
import statistics

from mandelbrot_parallel import mandelbrot_chunk, mandelbrot_serial

#lesson 5 milestone1 
def mandelbrot_dask(
    N, x_min, x_max, y_min, y_max,
    max_iter=100, n_chunks=32
):
    chunk_size = max(1, N // n_chunks)

    tasks = []
    row = 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(
            delayed(mandelbrot_chunk)(
                row, row_end, N, x_min, x_max, y_min, y_max, max_iter
            )
        )
        row = row_end

    parts = dask.compute(*tasks)
    return np.vstack(parts)


if __name__ == "__main__":
    N = 1024
    max_iter = 100
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.25, 1.25

    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)

    print(f"Dashboard: {client.dashboard_link}")

    client.run(lambda: mandelbrot_chunk(0, 8, 8, x_min, x_max, y_min, y_max, 10))

    result_serial = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
    result_dask = mandelbrot_dask(
        N, x_min, x_max, y_min, y_max,
        max_iter=max_iter,
        n_chunks=32
    )

    print("Dask matches serial:", np.array_equal(result_serial, result_dask))

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        result = mandelbrot_dask(
            N, x_min, x_max, y_min, y_max,
            max_iter=max_iter,
            n_chunks=32
        )
        times.append(time.perf_counter() - t0)

    print(f"Dask local (n_chunks=32): {statistics.median(times):.3f} s")

    client.close()
    cluster.close()