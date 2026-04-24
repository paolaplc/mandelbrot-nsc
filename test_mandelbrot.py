import pytest
import numpy as np

from mandelbrot import (
    mandelbrot_point,
    compute_mandelbrot_grid,
    compute_mandelbrot_vectorized,
    compute_mandelbrot_numba,
)


KNOWN_CASES = [
    (0 + 0j, 100, 100),
    (5.0 + 0j, 100, 1),
    (-2.5 + 0j, 100, 1),
    (0 + 2j, 100, 2),
]


def test_origin():
    assert mandelbrot_point(0 + 0j, 100) == 100


def test_far_outside():
    assert mandelbrot_point(5.0 + 0j, 100) == 1


def test_boundary_escape_condition():
    assert mandelbrot_point(0 + 2j, 100) == 2


@pytest.mark.parametrize("c, max_iter, expected", KNOWN_CASES)
def test_known_pixel_values(c, max_iter, expected):
    assert mandelbrot_point(c, max_iter) == expected


def test_numpy_matches_naive_small_grid():
    ref = compute_mandelbrot_grid(-2, 1, -1.5, 1.5, 32, 32, 50)
    test = compute_mandelbrot_vectorized(-2, 1, -1.5, 1.5, 32, 50)

    assert np.array_equal(ref, test)


def test_numba_matches_naive_small_grid():
    ref = compute_mandelbrot_grid(-2, 1, -1.5, 1.5, 32, 32, 50)

    compute_mandelbrot_numba(-2, 1, -1.5, 1.5, 8, 8, 10)

    test = compute_mandelbrot_numba(-2, 1, -1.5, 1.5, 32, 32, 50)

    assert np.array_equal(ref, test)

    