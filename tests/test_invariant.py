import numpy as np
from numpy2ad import transform
from pytest import approx
from typing import Callable


def serialize(*args: np.ndarray):
    return np.concatenate([x.ravel() for x in list(args)])


def central_fd(func: Callable, *args: np.ndarray, wrt: int, index: int) -> float:
    """Computes the central finite difference approximation (2nd order)
    for the derivative of func with respect to the specified input and index.

    Args:
        func (Callable): function with signature f(*args) -> np.ndarray
        wrt (int): input in *args to differentiate by
        index (int): entry in input to differentiate by

    Returns:
        float: the finite difference approximation
    """
    inputs_copy = [i.copy() for i in list(args)]

    def round_to_binary(h):
        return np.power(2.0, np.round(np.log(h) / np.log(2.0)))

    value = (inputs_copy[wrt]).flat[index].copy()
    h = round_to_binary(
        np.cbrt(np.finfo(np.float64).eps) * (1.0 + np.abs(value))
    )  # ~7.63e-06

    (inputs_copy[wrt]).flat[index] = value - h
    y0 = func(*inputs_copy)

    (inputs_copy[wrt]).flat[index] = value + h
    y1 = func(*inputs_copy)

    return (y1 - y0) / (2 * h)


def test_central_fd():
    def f_sq(x: np.ndarray):
        return x**2

    x = np.ones(10)

    dy = central_fd(f_sq, x, wrt=0, index=0)
    assert dy[0] == approx(2.0)

    def f_add(A: np.ndarray, B: np.ndarray):
        return A + B

    A = np.random.rand(5, 5)
    B = np.random.rand(5, 5)

    # df / dA[0, 0] -> first row in f affected
    dA = central_fd(f_add, A, B, wrt=0, index=0)
    assert dA[0, 0] == 1.0
    assert np.count_nonzero(dA) == 1

    dB = central_fd(f_add, A, B, wrt=1, index=24)
    assert dB[4, 4] == 1.0
    assert np.count_nonzero(dB) == 1

    def f_mul(A: np.ndarray, B: np.ndarray):
        return A @ B

    A = np.random.rand(10, 10)
    B = np.random.rand(10, 10)

    dA = central_fd(f_mul, A, B, wrt=0, index=0)  # perturb first entry in A
    assert dA[0, :] == approx(B[0, :])  # first row in f
    assert np.count_nonzero(dA) == 10

    dB = central_fd(f_mul, A, B, wrt=1, index=0)
    assert dB[:, 0] == approx(A[:, 0])  # first column in f
    assert np.count_nonzero(dB) == 10


def random_invertible(shape: tuple, method="det") -> np.ndarray:
    """Returns a uniformly random (in (0, 1]) and diagonally dominant matrix of given shape.

    Args:
        shape (tuple): shape of matrix (vectors are one-column matrices)

    Returns:
        np.ndarray: generated matrix
    """
    M = (
        np.random.default_rng()
        .uniform(low=0.0, high=1.0, size=shape)
        .astype(dtype=np.float64)
    )
    if len(shape) > 1:
        if shape[0] == shape[1]:  # square matrix
            if method == "diagonal":  # normalized diagonal dominance -> ensures regularity
                row_sum = np.sum(np.abs(M), axis=1)
                np.fill_diagonal(M, row_sum)
                M /= np.max(M)
            elif method == "symm":  # symmetric -> det(M) too small
                M = 0.5 * (M + M.T)
            elif method == "det":  # unit determinant -> invariants too small?
                det = np.linalg.det(M)
                M = M * np.sign(det) * (1.0 / np.abs(det)) ** (1.0 / shape[0])
            elif method == "rand":
                pass
            else:
                raise ValueError("method not recognized")

        # TODO: what about rectangular (m > n) matrices? A @ A.T becomes singular!

    return M


def random_inputs_invariants(func: Callable, out_shape: tuple, *input_shapes: tuple):
    """Tests func for differential invariant with random input matrices

    Args:
        func (Callable): f: *input_shapes -> out_shape
        out_shape (tuple): shape of output of f (used for out_a)
    """
    # transform function and make it visible
    exec(compile(transform(func), filename="<ast>", mode="exec"), globals())
    func_ad = eval(func.__name__ + "_ad")

    num_out_adj = np.prod(out_shape)
    for i in range(num_out_adj):
        # random inputs for i-th output adjoint
        inputs = [
            random_invertible(shape, method="diagonal") for shape in list(input_shapes)
        ]
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
                Y_T = serialize(central_fd(func, *inputs, wrt=wrt, index=index_x))
                X_A = serialize(*result)
                Y_A = serialize(out_a)

                # differential invariant
                adj_inv = X_A @ X_T
                tan_inv = Y_A @ Y_T
                abs_error = abs(adj_inv - tan_inv)

                if adj_inv > np.finfo(np.float64).eps:
                    rel_error = abs_error / abs(adj_inv)
                    rel_tol = np.cbrt(np.finfo(np.float64).eps)  # 6.055e-06
                    assert (
                        rel_error < rel_tol or abs_error < 1e-9
                    )  # TODO: what is a reasonable abs tol?


def test_mma():
    # Matrix multiply-add
    def mma(A, B, C):
        return A @ B + C

    random_inputs_invariants(mma, (3, 3), (3, 3), (3, 3), (3, 3))
    random_inputs_invariants(mma, (10, 10), (10, 10), (10, 10), (10, 10))
    random_inputs_invariants(mma, (5, 5), (5, 15), (15, 5), (5, 5))


def test_inv_scale():
    def inv_scale(A: np.ndarray, k: np.ndarray):
        scale = 1.0 / k
        return scale * A  # element-wise

    random_inputs_invariants(inv_scale, (5, 5), (5, 5), (5, 5))


def test_quadric():
    # requires second order accuracy!
    def quadric(A, B, C, D):
        # (n,m)(m,m)(m,n) + (n,m)(m,n) + (n,n)
        return B.T @ A @ B + B.T @ C + D

    shape_A = (15, 15)
    shape_B = (15, 10)
    shape_C = (15, 10)
    shape_D = (10, 10)

    random_inputs_invariants(quadric, shape_D, shape_A, shape_B, shape_C, shape_D)


def test_inverse():
    def inverse(A):
        return np.linalg.inv(A)

    random_inputs_invariants(inverse, (5, 5), (5, 5))
    random_inputs_invariants(inverse, (10, 10), (10, 10))
    random_inputs_invariants(inverse, (15, 15), (15, 15))


def test_gls():
    def GLS(M, X, y):
        # M: n x n, X: n x m, y: n x 1
        M_inv = np.linalg.inv(M)
        return np.linalg.inv(X.T @ M_inv @ X) @ X.T @ M_inv @ y

    dims_M = (10, 10)
    dims_X = (10, 5)
    dims_y = (10, 1)
    dims_b = (5, 1)

    random_inputs_invariants(GLS, dims_b, dims_M, dims_X, dims_y)
    random_inputs_invariants(GLS, (10, 1), (20, 20), (20, 10), (20, 1))


# A.2
def test_Optimization():
    def x_f(W, A, b, x):
        # W: n x n (diag spd), A: m x n, b: m x 1, x: n x 1, out: n x 1
        return W @ A.T @ np.linalg.inv(A @ W @ A.T) @ (b - A @ x)

    def x_o(W, A, x, c):
        # c: n x 1
        return W @ (A.T @ np.linalg.inv(A @ W @ A.T) @ A @ x - c)

    # dims_W = (5, 5)
    dims_W = (10, 10)
    # dims_A = (10, 5)
    dims_A = (10, 10)
    # dims_b = (10, 1)
    dims_b = (10, 1)
    # dims_x = (5, 1)
    dims_x = (10, 1)
    # dims_c = (5, 1)
    dims_c = (10, 1)
    # dims_out = (5, 1)
    dims_out = dims_x

    random_inputs_invariants(x_f, dims_out, dims_W, dims_A, dims_b, dims_x)
    random_inputs_invariants(x_o, dims_out, dims_W, dims_A, dims_x, dims_c)


# A.3
def test_Signal_Processing():
    pass


if __name__ == "__main__":
    test_central_fd()
    test_mma()
    test_inv_scale()
    test_quadric()
    test_inverse()
    test_gls()
    test_Optimization()
