import numpy as np
from dask import delayed
from dask.distributed import Client, LocalCluster #local 
#from dask.distributed import Client #strato 
import dask
import time
import statistics
import matplotlib.pyplot as plt


from mandelbrot_parallel import mandelbrot_chunk, mandelbrot_serial

#lesson 5 milestone1 
def mandelbrot_dask(
    N, x_min, x_max, y_min, y_max,
    max_iter=100, n_chunks=32
):
    """Compute Mandelbrot set using Dask."""
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
    N, max_iter = 4096, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    n_workers = 8


    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1) #local 
    client = Client(cluster) #local 
    #client = Client("tcp://10.92.1.19:8786") #Strato 

    print(f"Dashboard: {client.dashboard_link}")

    # warm up all workers
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))

    # M1: verify against serial
    ref = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    result = mandelbrot_dask(
        N, X_MIN, X_MAX, Y_MIN, Y_MAX,
        max_iter=max_iter,
        n_chunks=32
    )
    print("Dask matches serial:", np.array_equal(ref, result))

    # M1: benchmark n_chunks=32
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        result = mandelbrot_dask(
            N, X_MIN, X_MAX, Y_MIN, Y_MAX,
            max_iter=max_iter,
            n_chunks=32
        )
        times.append(time.perf_counter() - t0)

    print(f"Dask local (n_chunks=32): {statistics.median(times):.3f} s")
    #print(f"Dask cluster (n_chunks=32): {statistics.median(times):.3f} s")

    # M2: serial baseline
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)

    t_serial = statistics.median(times)
    print(f"Serial: {t_serial:.3f}s")

    # M2: chunk sweep
    chunk_results = []

    for mult in [1, 2, 4, 8, 16]:
        n_chunks = mult * n_workers

        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            mandelbrot_dask(
                N, X_MIN, X_MAX, Y_MIN, Y_MAX,
                max_iter=max_iter,
                n_chunks=n_chunks
            )
            times.append(time.perf_counter() - t0)

        t_par = statistics.median(times)
        lif = n_workers * t_par / t_serial - 1
        speedup = t_serial / t_par

        chunk_results.append((n_chunks, t_par, speedup, lif))

        print(f"{n_chunks:4d} chunks {t_par:.3f}s {speedup:.1f}x LIF={lif:.2f}")

    chunks = [r[0] for r in chunk_results]
    times_plot = [r[1] for r in chunk_results]

    plt.figure()
    plt.plot(chunks, times_plot, marker="o")
    plt.xlabel("Number of chunks")
    plt.ylabel("Runtime (s)")
    plt.title("Dask local chunk sweep")
    #plt.title("Dask cluster chunk sweep")
    plt.xscale("log")
    plt.grid(True)
    plt.savefig("dask_chunk_sweep.png", dpi=150)
    #plt.savefig("dask_cluster_chunk_sweep.png", dpi=150)
    plt.show()

    client.close()
    cluster.close()