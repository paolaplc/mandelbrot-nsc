import numpy as np
import matplotlib.pyplot as plt

#Profiling, row vs col, scaling, plots


from mandelbrot import (
    create_complex_grid,
    mandelbrot_point,
    compute_mandelbrot_grid,
    compute_mandelbrot_vectorized,
    compute_mandelbrot_numba,
    benchmark,
    profile_function
)



#l2:milestone3 : memory access patterns 
def row_sum(A):
    N = A.shape[0]
    s = 0.0
    for i in range(N):
        s += np.sum(A[i, :])
    return s

def col_sum(A):
    N = A.shape[1]
    s = 0.0
    for j in range(N):
        s += np.sum(A[:, j])
    return s




# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    print("RUNNING mandelbrot.py")
    #lesson2 milestone 1
    C = create_complex_grid(-2, 1, -1.5, 1.5, 1024)
    print(f"Shape: {C.shape}")
    print(f"Type:  {C.dtype}")

    # sanity check
    print("mandelbrot_point(0) =", mandelbrot_point(0))

    
    #lessin 3 milestone 1 calling profiling 
    print("\n--- cProfile: Naive 512x512 ---")
    profile_function(
        lambda: compute_mandelbrot_grid(-2, 1, -1.5, 1.5, 512, 512, 100),
        "naive_profile.prof",
        top_n=10
    )
    

    print("\n--- cProfile: Vectorized NumPy 512x512 ---")
    profile_function(
        lambda: compute_mandelbrot_vectorized(-2, 1, -1.5, 1.5, 512, 100),
        "numpy_profile.prof",
        top_n=10
    )
    

    # benchmark naive baseline (>=3 runs, perf_counter)
    t_med, result_naive = benchmark(
        compute_mandelbrot_grid,
        -2, 1, -1.5, 1.5,
        1024, 1024, 100,
        n_runs=3
    )

    print("Result shape:", result.shape)

    print("\nVectorized:")
    t_vec, result_vec = benchmark(compute_mandelbrot_vectorized, -2, 1, -1.5, 1.5, 1024, 100, n_runs=3)
    print(result_vec.shape)

    # validating (slide version)
    if np.allclose(result_naive, result_vec):
        print("Results match!")
    else:
        print("Results differ!")

    diff = np.abs(result_naive - result_vec)
    print(f"Max difference: {diff.max()}")
    print(f"Different pixels: {(diff > 0).sum()}")

    print(f"Speedup: {t_med / t_vec:.2f}x")

    # plot 
    plt.imshow(result_vec, cmap="hot", origin="lower")
    plt.colorbar(label="Iteration count")
    plt.title(f"Mandelbrot Set (vectorized) median={t_vec:.3f}s")
    plt.savefig("mandelbrot_vectorized.png", dpi=150)
    plt.show()

    plt.imshow(result_naive, cmap="hot", origin="lower")
    plt.colorbar(label="Iteration count")
    plt.title(f"Mandelbrot Set (naive) median={t_med:.3f}s")
    plt.savefig("mandelbrot_naive.png", dpi=150)
    plt.show()

        
    #milestone3 memory access patterns
    N = 10000
    A = np.random.rand(N, N)

    print("C-order:", A.flags["C_CONTIGUOUS"], "F-order:", A.flags["F_CONTIGUOUS"])

    t_row, _ = benchmark(row_sum, A, n_runs=3)
    t_col, _ = benchmark(col_sum, A, n_runs=3)

    print("C-order times -> row:", t_row, "col:", t_col)

    A_f = np.asfortranarray(A)
    print("C-order:", A_f.flags["C_CONTIGUOUS"], "F-order:", A_f.flags["F_CONTIGUOUS"])

    t_row_f, _ = benchmark(row_sum, A_f, n_runs=3)
    t_col_f, _ = benchmark(col_sum, A_f, n_runs=3)

    print("F-order times -> row:", t_row_f, "col:", t_col_f)

    #l2:molestone4: Scaling 

    sizes = [256, 512, 1024, 2048, 4096]
    runtimes = []

    for n in sizes:
        t_n, _ = benchmark(
            compute_mandelbrot_vectorized,
            -2, 1, -1.5, 1.5,
            n, 100,
            n_runs=3
        )
        runtimes.append(t_n)

    # plot: grid size vs runtime
    plt.figure()
    plt.plot(sizes, runtimes, marker="o")
    plt.xlabel("Grid size n (n×n)")
    plt.ylabel("Runtime (median seconds)")
    plt.title("Vectorized Mandelbrot: runtime vs grid size")
    plt.grid(True)
    plt.show()

    #predicting 2048^2 from 1024^2
    X = runtimes[sizes.index(1024)]
    pred_2048 = 4 * X
    meas_2048 = runtimes[sizes.index(2048)]
    print(f"Prediction: if 1024x1024 takes X={X:.4f}s, then 2048x2048 ~ 4X = {pred_2048:.4f}s")
    print(f"Measured 2048x2048: {meas_2048:.4f}s   Ratio(measured/pred)={meas_2048/pred_2048:.3f}")




