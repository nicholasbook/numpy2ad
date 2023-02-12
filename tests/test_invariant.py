# Differential invariant: x_a x_t = y_a y_t (first order)
# y = F(x), F: R^N -> R^M, F' € R^MxN
# x_a € R^1xN (row of F'), x_t € R^N (seed), y_a € R^M (seed), y_t € R^N (column of F')

import numpy as np
import numpy2ad
import math
from pytest import approx


def func_fma(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    return A @ B + C


def tangent_fd(f, x):
    """tangent approximated by finite difference"""
    h = math.sqrt(math.ulp(x))
    return 1 / h * (f(x + h / 2) - f(x - h / 2))


def test_fd():
    assert tangent_fd(lambda x: x**2, 1) == approx(2.0)
    assert tangent_fd(lambda x: x**2, 1000) == approx(2000.0)
    assert tangent_fd(lambda x: x**2, 100000) == approx(200000.0)
    # fails for 1e6


def test_func():
    # make func_fma_ad visible
    exec(compile(numpy2ad.transform(func_fma), filename="<ast>", mode="exec"))
