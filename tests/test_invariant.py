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


# def num_entries_in(*args: np.ndarray):
#     return np.sum([np.prod(input.shape) for input in list(args)])


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
    h = np.sqrt(np.finfo(float).eps)

    x0 = [x.copy() for x in args]
    (x0[wrt]).flat[index] += h / 2
    x1 = [x.copy() for x in args]
    (x1[wrt]).flat[index] -= h / 2

    return 1 / h * (func(*x0) - func(*x1))


def central_fd_no_copy(func: Callable, *args: np.ndarray, wrt: int, index: int):
    """First-order central finite difference with 'flat' indexing

    Args:
        func: function to differentiate
        *args (np.ndarray): arguments of func
        wrt (int): which argument to func
        index (int): which entry of argument
    """
    h = np.sqrt(np.finfo(float).eps)

    x = list(args)
    (x[wrt]).flat[index] += h / 2
    y0 = func(*x)

    (x[wrt]).flat[index] -= h
    y1 = func(*x)

    return 1 / h * (y0 - y1)


def test_central_fd():
    def f(x: np.ndarray):
        return x**2

    x = np.ones(10)
    y = f(x)

    dy_slow = central_finite_diff(f, x, wrt=0, index=(0,))
    assert dy_slow[0] == approx(2.0)

    dy_lin = central_fd_flat(f, x, wrt=0, index=0)
    assert dy_lin[0] == approx(2.0)

    dy_fast = central_fd_no_copy(f, x, wrt=0, index=0)
    assert dy_fast[0] == approx(2.0)


def mma(A, B, C):  # Matrix-matrix multiply and add
    return A @ B + C


def test_mma(tmp_pkg):
    # generate adjoint code

    transform(mma, pathlib.Path(tmp_pkg) / "out.py")
    from out import mma_ad

    dims = (10, 10)
    A = np.random.rand(*dims)
    B = np.random.rand(*dims)
    C = np.random.rand(*dims)
    # initialize adjoints
    A_a = np.zeros(dims)
    B_a = np.zeros(dims)
    C_a = np.zeros(dims)
    Y_a = np.zeros(dims)

    # seed the adjoint
    direction = (0, 0)
    Y_a[*direction] = 1

    # adjoint model
    Y, A_a, B_a, C_a = mma_ad(A, B, C, A_a, B_a, C_a, Y_a)

    X_A = serialize(A_a, B_a, C_a)
    Y_A = serialize(Y_a)

    for wrt in [0, 1, 2]:
        # tangent approximation of i-th matrix
        x_t = np.zeros(dims)
        x_t[*direction] = 1.0
        X = [np.zeros(dims), np.zeros(dims), np.zeros(dims)]
        X[wrt] = x_t
        X_T = serialize(*X)
        Y_T = serialize(central_finite_diff(mma, A, B, C, wrt=wrt, index=direction))

        assert X_A @ X_T == approx(Y_A @ Y_T)


def quadric(A, B, C, D):  # (n,m)(m,m)(m,n) + (n,m)(m,n) + (n,n)
    return B.T @ A @ B + B.T @ C + D


def test_quadric(tmp_pkg):
    shape = (15, 10)
    A = np.random.rand(shape[0], shape[0])
    B = np.random.rand(*shape)
    C = np.random.rand(*shape)
    D = np.random.rand(shape[1], shape[1])

    transform(quadric, pathlib.Path(tmp_pkg) / "out2.py")
    from out2 import quadric_ad

    A_a = np.zeros(A.shape)
    B_a = np.zeros(B.shape)
    C_a = np.zeros(C.shape)
    D_a = np.zeros(D.shape)
    Y_a = np.zeros(D.shape)
    direction = (9, 9)  # last entry
    Y_a[*direction] = 1  # seed adjoint

    Y, A_a, B_a, C_a, D_a = quadric_ad(A, B, C, D, A_a, B_a, C_a, D_a, Y_a)
    X_A = serialize(A_a, B_a, C_a, D_a)
    Y_A = serialize(Y_a)

    for wrt, x in enumerate([A, B, C, D]):
        # tangent approximation of i-th matrix
        direction = (x.shape[0] - 1, x.shape[1] - 1)  # last entry
        x_t = np.zeros(x.shape)
        x_t[*direction] = 1.0
        X = [np.zeros(A.shape), np.zeros(B.shape), np.zeros(C.shape), np.zeros(D.shape)]
        X[wrt] = x_t
        X_T = serialize(*X)

        Y_T = serialize(
            central_finite_diff(quadric, A, B, C, D, wrt=wrt, index=direction)
        )

        assert X_A @ X_T == approx(Y_A @ Y_T, abs=1e-5)


def inverse(A):
    return np.linalg.inv(A)


def test_inverse(tmp_pkg):
    dims = (10, 10)
    A = np.random.rand(*dims) + 1e-4

    transform(inverse, pathlib.Path(tmp_pkg) / "out3.py")
    from out3 import inverse_ad

    num_tests = 10

    for i in range(num_tests):
        # seed random entry in Y
        m = np.random.randint(0, dims[0] - 1)
        n = np.random.randint(0, dims[1] - 1)
        Y_a = np.zeros(dims)
        Y_a[m, n] = 1.0

        A_a = np.zeros(A.shape)
        _, A_a = inverse_ad(A, A_a, Y_a)

        for j in range(num_tests):
            A_t = np.zeros(A.shape)
            mj = np.random.randint(0, dims[0] - 1)
            nj = np.random.randint(0, dims[1] - 1)
            A_t[mj, nj] = 1.0  # permute random entry in A

            Y_T = serialize(central_finite_diff(inverse, A, wrt=0, index=(mj, nj)))
            X_T = serialize(A_t)
            X_A = serialize(A_a)
            Y_A = serialize(Y_a)

            # with absolute tolerance
            assert X_A @ X_T == approx(Y_A @ Y_T, abs=1e-4)


def test_GLS():
    def GLS(M, X, y):
        M_inv = np.linalg.inv(M)
        return np.linalg.inv(X.T @ M_inv @ X) @ X.T @ M_inv @ y

    # generate GLS_ad
    exec(compile(transform(GLS), filename="<ast>", mode="exec"), globals())

    dims_M = (20, 20)
    dims_X = (20, 5)
    dims_y = (20, 1)
    dims_b = (5, 1)

    for index_b in range(dims_b[0]):
        b_a = np.zeros(dims_b)
        b_a.flat[index_b] = 1.0

        # create new random model
        M = np.random.rand(*dims_M) + 1e-4
        M_a = np.zeros_like(M)
        X = np.random.rand(*dims_X)
        X_a = np.zeros_like(X)
        y = np.random.rand(*dims_y)
        y_a = np.zeros_like(y)

        _, M_a, X_a, y_a = GLS_ad(M, X, y, M_a, X_a, y_a, out_a=b_a)

        inputs = [M, X, y]
        input_adjoints = [M_a, X_a, y_a]

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
                Y_T = serialize(central_fd_flat(GLS, *inputs, wrt=wrt, index=index_x))
                X_A = serialize(*input_adjoints)
                Y_A = serialize(b_a)

                assert X_A @ X_T == approx(Y_A @ Y_T, abs=1e-3)


def inv_scale(A, k):
    scale = 1.0 / k
    return scale * A  # element-wise


def test_inv_scale(tmp_pkg):
    dims = (5, 5)
    A = np.random.rand(*dims)
    k = np.random.rand(*dims) + 1e-4

    transform(inv_scale, pathlib.Path(tmp_pkg) / "out_inv_scale.py")
    from out_inv_scale import inv_scale_ad

    num_tests = 25
    for _ in range(num_tests):
        # seed random entry in out_a
        m = np.random.randint(0, dims[0] - 1)
        n = np.random.randint(0, dims[1] - 1)
        out_a = np.zeros(dims)
        out_a[m, n] = 1.0

        # adjoint model
        A_a = np.zeros_like(A)
        k_a = np.zeros_like(k)
        _, A_a, k_a = inv_scale_ad(A, k, A_a, k_a, out_a)

        for wrt, x in enumerate([A, k]):
            for i in range(x.size):
                # seed tangent for i-th entry
                x_t = np.zeros_like(x)
                x_t.flat[i] = 1.0

                X = [np.zeros_like(A), np.zeros_like(k)]  # TODO: generalize
                X[wrt] = x_t
                X_T = serialize(*X)
                Y_T = serialize(
                    central_finite_diff(
                        inv_scale, A, k, wrt=wrt, index=np.unravel_index(i, dims)
                    )
                )
                X_A = serialize(A_a, k_a)
                Y_A = serialize(out_a)

                assert X_A @ X_T == approx(Y_A @ Y_T)


if __name__ == "__main__":
    # create tmp directory
    tmp = pathlib.Path("tmp")
    if not tmp.exists():
        tmp.mkdir()
    f = open(tmp / "__init__.py", "w")
    f.close()
    sys.path.insert(0, "tmp")

    test_central_fd()
    test_mma(tmp)
    test_quadric(tmp)
    test_inverse(tmp)
    test_GLS()
    test_inv_scale(tmp)

    if tmp.exists():
        shutil.rmtree(tmp)
