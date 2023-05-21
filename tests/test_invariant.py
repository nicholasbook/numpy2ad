import numpy
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
        wrt (int): which argument to func
        index (tuple): which entry of argument
        *args (np.ndarray): arguments of func
    """
    h = np.sqrt(np.finfo(float).eps)
    x0 = [x.copy() for x in args]
    x0[wrt][*index] += h / 2
    x1 = [x.copy() for x in args]
    x1[wrt][*index] -= h / 2
    return 1 / h * (func(*x0) - func(*x1))


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
    return numpy.linalg.inv(A)


def test_inverse(tmp_pkg):
    dims = (10, 10)
    A = np.random.rand(*dims)

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


def GLS(M, X, y):
    M_inv = numpy.linalg.inv(M)
    return numpy.linalg.inv(X.T @ M_inv @ X) @ X.T @ M_inv @ y


def test_GLS():
    # TODO
    pass


if __name__ == "__main__":
    # create tmp directory
    tmp = pathlib.Path("tmp")
    if not os.path.isdir(tmp):
        os.mkdir(tmp)
    f = open(tmp / "__init__.py", "w")
    f.close()
    sys.path.insert(0, "tmp")

    test_mma(tmp)
    test_quadric(tmp)
    test_inverse(tmp)
    test_GLS()

    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
