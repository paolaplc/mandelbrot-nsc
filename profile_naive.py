
from mandelbrot import compute_mandelbrot_grid

try:
    profile
except NameError:
    def profile(func):
        return func


@profile
def run():
    compute_mandelbrot_grid(-2, 1, -1.5, 1.5, 512, 512, 100)


if __name__ == "__main__":
    run()

