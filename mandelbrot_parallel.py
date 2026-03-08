import numpy as np
from numba import njit
from mandelbrot import compute_mandelbrot_numba


#lesson4 milestone 1
@njit
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = 0.0
    z_imag = 0.0

    for i in range(max_iter):
        zr2 = z_real * z_real
        zi2 = z_imag * z_imag

        if zr2 + zi2 > 4.0:
            return i

        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = zr2 - zi2 + c_real

    return max_iter


@njit
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)

    dx = (x_max - x_min) / (N - 1)
    dy = (y_max - y_min) / (N - 1)

    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy

        for col in range(N):
            c_real = x_min + col * dx
            out[r, col] = mandelbrot_pixel(c_real, c_imag, max_iter)

    return out


def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)




if __name__ == "__main__":
    N = 1024
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.25, 1.25
    max_iter = 100

    result = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)

    result_old = compute_mandelbrot_numba(-2.5, 1.0, -1.25, 1.25, 1024, 1024, 100)
    
    #checking if match the numba l3 
    print("Match with L3 numba:", np.allclose(result, result_old))
    diff = np.abs(result - result_old)
    print("Max difference:", diff.max())
    print("Different pixels:", (diff > 0).sum())

    