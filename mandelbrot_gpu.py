import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
import time

KERNEL_SRC = """
__kernel void mandelbrot(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter
)
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


KERNEL_SRC_F64 = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void mandelbrot_f64(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter
)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (col >= N || row >= N) return;

    double c_real = x_min + col * (x_max - x_min) / (double)(N - 1);
    double c_imag = y_min + row * (y_max - y_min) / (double)(N - 1);

    double zr = 0.0;
    double zi = 0.0;

    int count = 0;

    while (count < max_iter && zr*zr + zi*zi <= 4.0) {
        double tmp = zr*zr - zi*zi + c_real;
        zi = 2.0 * zr * zi + c_imag;
        zr = tmp;
        count++;
    }

    result[row * N + col] = count;
}
"""



ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
prog = cl.Program(ctx, KERNEL_SRC).build()

#prog_f64 = cl.Program(ctx, KERNEL_SRC_F64).build()

dev = ctx.devices[0]
print("Device:", dev.name)

supports_fp64 = "cl_khr_fp64" in dev.extensions

if not supports_fp64:
    print("No native fp64 -- f64 OpenCL kernel skipped")
else:
    print("Native fp64 supported")
    prog_f64 = cl.Program(ctx, KERNEL_SRC_F64).build()

N = 1024
MAX_ITER = 200

image = np.zeros((N, N), dtype=np.int32)
image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)

X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25

# warm-up
prog.mandelbrot(
    queue, (64, 64), None,
    image_dev,
    np.float32(X_MIN), np.float32(X_MAX),
    np.float32(Y_MIN), np.float32(Y_MAX),
    np.int32(64), np.int32(MAX_ITER),
)
queue.finish()

# timed run
t0 = time.perf_counter()

prog.mandelbrot(
    queue, (N, N), None,
    image_dev,
    np.float32(X_MIN), np.float32(X_MAX),
    np.float32(Y_MIN), np.float32(Y_MAX),
    np.int32(N), np.int32(MAX_ITER),
)

queue.finish()
elapsed = time.perf_counter() - t0


print(f"GPU {N}x{N}: {elapsed * 1000:.2f} ms")
cl.enqueue_copy(queue, image, image_dev)
queue.finish()


# time f64 
if supports_fp64:
    t0 = time.perf_counter()

    prog_f64.mandelbrot_f64(
        queue, (N, N), None,
        image_dev,
        np.float64(X_MIN), np.float64(X_MAX),
        np.float64(Y_MIN), np.float64(Y_MAX),
        np.int32(N), np.int32(MAX_ITER),
    )

    queue.finish()
    elapsed_f64 = time.perf_counter() - t0

    print(f"GPU f64 {N}x{N}: {elapsed_f64 * 1000:.2f} ms")
    print(f"f64/f32 slowdown: {elapsed_f64 / elapsed:.2f}x")
else:
    print("GPU f64 skipped: cl_khr_fp64 not supported on this device")

plt.imshow(image, cmap="hot", origin="lower")
plt.axis("off")
plt.savefig("mandelbrot_gpu.png", dpi=150, bbox_inches="tight")
plt.show()