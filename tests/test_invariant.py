import numpy as np
from numpy2ad import transform
import pytest
from pytest import approx
import os
import sys
import pathlib
import shutil
from typing import Callable


@pytest.fixture(scope="session")
def tmp_pkg(tmp_path_factory):
    init = tmp_path_factory.mktemp("tmp_pkg") / "__init__.py"
    sys.path.insert(0, str(init.parent))
    return str(init.parent)


def serialize(*args: np.ndarray):
    return np.concatenate([x.ravel() for x in list(args)])


def central_finite_diff(func: Callable, *args: np.ndarray, wrt: int, index: tuple):
    """First-order central finite difference

    Args:
        func: function to differentiate
        *args (np.ndarray): arguments of func
        wrt (int): which argument to func
        index (tuple): which entry of argument
    """
    h = np.sqrt(np.finfo(float).eps)
    x0 = [x.copy() for x in args]
    x0[wrt][*index] += h / 2
    x1 = [x.copy() for x in args]
    x1[wrt][*index] -= h / 2
    return 1 / h * (func(*x0) - func(*x1))


def central_fd_flat(func: Callable, *args: np.ndarray, wrt: int, index: int):
    """First-order central finite difference with 'flat' indexing

    Args:
        func: function to differentiate
        *args (np.ndarray): arguments of func
        wrt (int): which argument to func
        index (int): which entry of argument
    """
    h = np.sqrt(np.finfo(np.float64).eps)

    x0 = [x.copy() for x in args]
    (x0[wrt]).flat[index] += h / 2
    x1 = [x.copy() for x in args]
    (x1[wrt]).flat[index] -= h / 2

    return 1 / h * (func(*x0) - func(*x1))


def central_fd_no_copy(func: Callable, *args: np.ndarray, wrt: int, index: int):
    """First-order central finite difference with 'flat' indexing and no copies.

    Args:
        func: function to differentiate
        *args (np.ndarray): arguments of func
        wrt (int): which argument to func
        index (int): which entry of argument
    """
    h = np.sqrt(np.finfo(float).eps)  # ~ 1e-8

    x = list(args)
    (x[wrt]).flat[index] += h / 2
    y0 = func(*x)

    (x[wrt]).flat[index] -= h
    y1 = func(*x)

    return 1 / h * (y0 - y1)


def central_fd_4th_order(func: Callable, *args: np.ndarray, wrt: int, index: int):
    """Fourth-order central finite difference.

    Args:
        func: function to differentiate
        *args (np.ndarray): arguments of func
        wrt (int): which argument to func
        index (int): which entry of argument
    """
    h = np.sqrt(np.finfo(float).eps)  # ~ 1e-8

    x = list(args)

    (x[wrt]).flat[index] += h
    y1 = func(*x)
    (x[wrt]).flat[index] -= h

    (x[wrt]).flat[index] += 2 * h
    y2 = func(*x)
    (x[wrt]).flat[index] -= 2 * h

    (x[wrt]).flat[index] -= h
    y_1 = func(*x)
    (x[wrt]).flat[index] += h

    (x[wrt]).flat[index] -= 2 * h
    y_2 = func(*x)
    (x[wrt]).flat[index] += 2 * h

    return 1 / h * (2 / 3 * y1 - 1 / 12 * y2 - 2 / 3 * y_1 + 1 / 12 * y_2)


def test_central_fd():
    def f_sq(x: np.ndarray):
        return x**2

    def f_mul(A: np.ndarray, B: np.ndarray):
        return A @ B

    x = np.ones(10)

    dy_slow = central_finite_diff(f_sq, x, wrt=0, index=(0,))
    assert dy_slow[0] == approx(2.0)

    dy_lin = central_fd_flat(f_sq, x, wrt=0, index=0)
    assert dy_lin[0] == approx(2.0)

    dy_fast = central_fd_no_copy(f_sq, x, wrt=0, index=0)
    assert dy_fast[0] == approx(2.0)

    A = np.random.rand(10, 10) + 1e-3
    B = np.random.rand(10, 10) + 1e-3

    dy_slow = central_finite_diff(
        f_mul, A, B, wrt=0, index=(0, 0)
    )  # perturb first entry in A
    assert dy_slow[0, :] == approx(B[0, :], rel=1e-5)

    dy_lin = central_fd_flat(f_mul, A, B, wrt=0, index=0)
    assert dy_lin[0, :] == approx(B[0, :], rel=1e-5)

    dy_fast = central_fd_no_copy(f_mul, A, B, wrt=0, index=0)
    assert dy_fast[0, :] == approx(B[0, :], rel=1e-5)

    dy_4th = central_fd_4th_order(f_mul, A, B, wrt=0, index=0)
    # print(dy_fast[0, :] - B[0, :])
    # print(dy_4th[0, :] - B[0, :])
    assert dy_4th[0, :] == approx(B[0, :], rel=1e-5)


def random_invertible(shape: tuple) -> np.ndarray:
    """Returns a uniformly random (in (1, 10]) and diagonally dominant matrix of given shape.

    Args:
        shape (tuple): shape of matrix (vectors are one-column matrices)

    Returns:
        np.ndarray: generated matrix
    """
    M = (
        np.random.default_rng()
        .uniform(low=1.0, high=10.0, size=shape)
        .astype(dtype=np.float64)
    )  # lower bound at 1 for better accuracy

    if len(shape) > 1:
        if shape[0] == shape[1]:  # make square matrices diagonally dominant
            row_sum = np.sum(np.abs(M), axis=1)
            np.fill_diagonal(M, row_sum)

    return M


def random_inputs_invariants(
    func: Callable,
    out_shape: tuple,
    *input_shapes: tuple,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-6,
):
    """Tests func for differential invariant with random input matrices

    Args:
        func (Callable): f: *input_shapes -> out_shape
        out_shape (tuple): shape of output of f (used for out_a)
        rel_tol (float, optional): relative tolerance. Defaults to 1e-3.
        abs_tol (float, optional): absolute tolerance. Defaults to 1e-6.
    """
    # transform function and make it visible
    exec(compile(transform(func), filename="<ast>", mode="exec"), globals())
    func_ad = eval(func.__name__ + "_ad")

    num_out_adj = np.prod(out_shape)
    for i in range(num_out_adj):
        # random inputs for i-th output adjoint
        inputs = [random_invertible(shape) for shape in list(input_shapes)]
        inputs_a = [np.zeros_like(entry) for entry in inputs]

        out_a = np.zeros(out_shape)
        out_a.flat[i] = 1.0  # seed

        result: tuple = func_ad(*inputs, *inputs_a, out_a)
        result = result[1:]  # discard primal result

        # test against all tangents
        for wrt, x in enumerate(inputs):
            for index_x in range(x.size):
                # seed tangent for i-th entry of current input
                x_t = np.zeros_like(x)
                x_t.flat[index_x] = 1.0

                # remaining tangents are zero
                X = [np.zeros_like(arg) for arg in inputs]
                X[wrt] = x_t

                X_T = serialize(*X)
                Y_T = serialize(
                    central_fd_no_copy(func, *inputs, wrt=wrt, index=index_x)
                )
                X_A = serialize(*result)
                Y_A = serialize(out_a)

                # differential invariant
                assert X_A @ X_T == approx(Y_A @ Y_T, rel=rel_tol, abs=abs_tol)


def test_mma():
    # Matrix multiply-add
    def mma(A, B, C):
        return A @ B + C

    random_inputs_invariants(mma, (3, 3), (3, 3), (3, 3), (3, 3))
    random_inputs_invariants(mma, (10, 10), (10, 10), (10, 10), (10, 10))
    random_inputs_invariants(mma, (5, 5), (5, 15), (15, 5), (5, 5))


def test_inverse():
    def inverse(A):
        return np.linalg.inv(A)

    random_inputs_invariants(inverse, (5, 5), (5, 5))
    random_inputs_invariants(inverse, (10, 10), (10, 10))
    random_inputs_invariants(inverse, (15, 15), (15, 15))


def test_gls():
    def GLS(M, X, y):
        M_inv = np.linalg.inv(M)
        return np.linalg.inv(X.T @ M_inv @ X) @ X.T @ M_inv @ y

    dims_M = (10, 10)
    dims_X = (10, 5)
    dims_y = (10, 1)
    dims_b = (5, 1)

    random_inputs_invariants(GLS, dims_b, dims_M, dims_X, dims_y)
    random_inputs_invariants(GLS, (10, 1), (20, 20), (20, 10), (20, 1))


def test_inv_scale():
    def inv_scale(A: np.ndarray, k: np.ndarray):
        scale = 1.0 / k
        return scale * A  # element-wise

    random_inputs_invariants(inv_scale, (5, 5), (5, 5), (5, 5))


def test_quadric():
    def quadric(A, B, C, D):
        # (n,m)(m,m)(m,n) + (n,m)(m,n) + (n,n)
        return B.T @ A @ B + B.T @ C + D

    shape_A = (15, 15)
    shape_B = (15, 10)
    shape_C = (15, 10)
    shape_D = (10, 10)

    random_inputs_invariants(quadric, shape_D, shape_A, shape_B, shape_C, shape_D)


if __name__ == "__main__":
    test_central_fd()
    test_mma()
    test_inverse()
    test_gls()
    test_inv_scale()
    test_quadric()
