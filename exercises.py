import numpy as np 

#find machie epsilon 
def find_machine_epsilon(dtype = np.float64):
    eps = dtype(1.0)
    while dtype(1.0) + eps / dtype(2.0) != dtype(1.0):
        eps = eps / dtype(2.0)
    return eps

for dtype in [np.float16, np.float32, np.float64]:
    computed = find_machine_epsilon(dtype)
    reference = np.finfo(dtype).eps
    print(f"{dtype.__name__}:")
    print(f" Computed: {float(computed):.4e}")
    print(f" np.info: {float(reference):.4e}" )
    print()


#Catastrophic cancellation

#quadratic form 
def quadratic_naive(a, b, c):
    t = type(a)

    disc = t(np.sqrt(b*b - t(4)*a*c))
    x1 = (-b + disc) / (t(2)*a)
    x2 = (-b - disc) / (t(2)*a)
    return x1, x2

# x^2 - 10000.0001*x + 1 = 0 roots: x1 ~ 10000, x2 ~ 1e-4

for dtype in [np.float16, np.float32, np.float64]:
    a = dtype(1.0)
    b = dtype(-10000.0001)
    c = dtype(1.0)

    x1, x2 = quadratic_naive(a, b, c)
    print(f"{dtype.__name__}:")
    print(f" Naive: x1={float(x1):.4e}, x2={float(x2):.4e}")
    print()


def quadratic_stable(a, b, c):
    t = type(a)

    disc = t(np.sqrt(b*b - t(4)*a*c))
    if b > 0:
        x1 = (-b - disc) / (t(2)*a)
    else:
        x1 = (-b + disc) / (t(2)*a)
    x2 = c / (a * x1)
    return x1, x2

true_small = 1.0 / 10000.0001
for dtype in [np.float16, np.float32, np.float64]:
    a, b, c = dtype(1.0), dtype(-10000.0001), dtype(1.0)
    _, x2_naive = quadratic_naive(a, b, c)
    _, x2_stable = quadratic_stable(a, b, c)
    err_naive = abs(float(x2_naive) - true_small) / true_small
    err_stable = abs(float(x2_stable) - true_small) / true_small
    print(f"{dtype.__name__}: naive={err_naive:.2e} stable={err_stable:.2e}")


#error accumulation in summation

n_values = [10, 100, 1_000, 10_000, 100_000]
for dtype in [np.float32, np.float64]:
    print(f"\n{dtype.__name__}:")
    for n in n_values:
        total = dtype(0.0)
        for _ in range(n):
            total += dtype(0.1)
        expected = n * 0.1
        rel_error = abs(float(total) - expected) / expected
        print(f" n={n:>7d}: result={float(total):.10f} rel_error={rel_error:.2e}")