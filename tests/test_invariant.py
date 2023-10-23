import numpy as np
from numpy2ad import transform
from pytest import approx
from typing import Callable
import pathlib


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
    h = round_to_binary(np.cbrt(np.finfo(np.float64).eps) * (1.0 + np.abs(value)))  # ~7.63e-06

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
        shape (tuple): shape of matrix.

        Note that vectors must have shape (n, 1).
        For input shape (n, ) a constant full (n, n) matrix is returned (for element-wise ops).

    Returns:
        np.ndarray: generated matrix
    """
    # if len(shape) == 1:  # e.g. (5, ) -> (5, 5) constant matrix
    #     return np.full(shape=(shape[0], shape[0]), fill_value=np.random.rand())
    # else:
    M = np.random.default_rng().uniform(low=0.0, high=1.0, size=shape).astype(dtype=np.float64)
    if len(shape) > 1:
        if shape[0] == shape[1]:  # square matrix
            if method == "diag":  # normalized diagonal dominance -> ensures regularity
                row_sum = np.sum(np.abs(M), axis=1)
                np.fill_diagonal(M, row_sum)
                M /= np.max(M)
            elif method == "symm":  # symmetric -> det(M) too small
                M = 0.5 * (M + M.T)
            elif method == "det":  # unit determinant -> invariants too small?
                det = np.linalg.det(M)
                M = M * np.sign(det) * (1.0 / np.abs(det)) ** (1.0 / shape[0])
            elif method == "sum":
                M = M.T @ M + shape[0] * np.eye(*shape)
            elif method == "rand":
                pass
            else:
                raise ValueError("method not recognized")

    return M


def random_inputs_invariants(
    func: Callable, out_shape: tuple, *input_shapes: tuple, method="diag", debug=False
):
    """Tests func for differential invariant with random input matrices

    Args:
        func (Callable): f: *input_shapes -> out_shape
        out_shape (tuple): shape of output of f (used for out_a)
    """
    # transform function and make it visible
    if not debug:
        exec(compile(transform(func), filename="<ast>", mode="exec"), globals())
        func_ad = eval(func.__name__ + "_ad")
    else:
        tests = pathlib.Path(__file__).parent
        tmp_module = tests / "debug"
        if not tmp_module.exists():
            tmp_module.mkdir()
            init = tmp_module / "__init__.py"
            f = init.open("w")
            f.close()
        module = tmp_module / f"{func.__name__}.py"
        transform(func, output_file=module)
        exec(f"from debug.{func.__name__} import {func.__name__}_ad")
        func_ad = eval(func.__name__ + "_ad")

    num_out_adj = np.prod(out_shape)
    for i in range(num_out_adj):
        # random inputs for i-th output adjoint
        inputs = [random_invertible(shape, method=method) for shape in list(input_shapes)]
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
                        rel_error < rel_tol or abs_error < 1e-11
                    )  # TODO: what is a reasonable abs tol?


def test_mma():
    # Matrix multiply-add
    def mma(A, B, C):
        return A @ B + C

    random_inputs_invariants(mma, (3, 3), (3, 3), (3, 3), (3, 3))
    random_inputs_invariants(mma, (10, 10), (10, 10), (10, 10), (10, 10))
    random_inputs_invariants(mma, (5, 5), (5, 15), (15, 5), (5, 5))


def test_elementwise(debug=False):
    # we want to support elementwise matrix-matrix and vector-vector multiplications
    # as they occur in the form of regularization terms in matrix problems

    def ew_product(A: np.ndarray, B: np.ndarray):
        return A * B

    shape_A = (5, 5)
    shape_B = (5, 5)
    random_inputs_invariants(ew_product, shape_A, shape_A, shape_B, debug=debug)
    random_inputs_invariants(ew_product, (3, 1), (3, 1), (3, 1))
    random_inputs_invariants(ew_product, (3,), (3,), (3,))

    # might be useful
    def ew_square(A: np.ndarray):
        return A**2

    random_inputs_invariants(ew_square, (3, 3), (3, 3))
    random_inputs_invariants(ew_square, (5,), (5,))


def test_quadric():
    # requires second order accuracy!
    def quadric(A, B, C, D):
        # (n,m)(m,m)(m,n) + (n,m)(m,n) + (n,n)
        return B.T @ A @ B + B.T @ C + D

    shape_A = (5, 5)
    shape_B = (5, 3)
    shape_C = (5, 3)
    shape_D = (3, 3)

    random_inputs_invariants(quadric, shape_D, shape_A, shape_B, shape_C, shape_D)
    # random_inputs_invariants(quadric, (4, 4), (8, 8), (8, 4), (8, 4), (4, 4))


def test_inverse():
    def inverse(A):
        return np.linalg.inv(A)

    random_inputs_invariants(inverse, (3, 3), (3, 3))
    random_inputs_invariants(inverse, (5, 5), (5, 5))
    random_inputs_invariants(inverse, (6, 6), (6, 6))


def test_gls():
    def GLS(M, X, y):
        # M: n x n, X: n x m, y: n x 1, b: m x 1, n > m
        M_inv = np.linalg.inv(M)
        return np.linalg.inv(X.T @ M_inv @ X) @ X.T @ M_inv @ y

    dims_M = (5, 5)
    dims_X = (5, 3)
    dims_y = (5, 1)
    dims_b = (3, 1)

    random_inputs_invariants(GLS, dims_b, dims_M, dims_X, dims_y)
    random_inputs_invariants(GLS, (4, 1), (8, 8), (8, 4), (8, 1))


# A.2
def test_Optimization():
    def x_f(W, A, b, x):
        # W: n x n (diag spd), A: m x n, b: m x 1, x: n x 1, out: n x 1, n > m
        return W @ A.T @ np.linalg.inv(A @ W @ A.T) @ (b - A @ x)

    def x_o(W, A, x, c):
        # c: n x 1
        return W @ (A.T @ np.linalg.inv(A @ W @ A.T) @ A @ x - c)

    dims_W = (5, 5)
    dims_A = (3, 5)
    dims_b = (3, 1)
    dims_x = (5, 1)
    dims_c = (5, 1)
    dims_out = dims_x

    random_inputs_invariants(x_f, dims_out, dims_W, dims_A, dims_b, dims_x)
    # random_inputs_invariants(x_f, (7, 1), (7, 7), (4, 7), (4, 1), (7, 1))
    random_inputs_invariants(x_o, dims_out, dims_W, dims_A, dims_x, dims_c)
    # random_inputs_invariants(x_o, (7, 1), (7, 7), (4, 7), (7, 1), (7, 1))


# A.3
def test_Signal_Processing():
    def SP(A, B, R, L, y):
        # A: n x n, B: n x n, R: (n-1) x n UT, L: (n-1)x(n-1) Diag, y: n x 1
        Ainv = np.linalg.inv(A)
        return (
            np.linalg.inv(Ainv.T @ B.T @ B @ Ainv + R.T @ L @ R) @ Ainv.T @ B.T @ B @ Ainv @ y
        )

    dims_A = (4, 4)
    dims_B = dims_A
    dims_R = (3, 4)
    dims_L = (3, 3)
    dims_y = (4, 1)
    dims_out = dims_y

    random_inputs_invariants(SP, dims_out, dims_A, dims_B, dims_R, dims_L, dims_y)
    random_inputs_invariants(SP, (7, 1), (7, 7), (7, 7), (6, 7), (6, 6), (7, 1))


def test_Ensemble_Kalman():
    def EKF(X_b, B, H, R, Y):
        # X_b: n x m, B: n x n, H: n x n, R: n x n spsd, Y: n x m, n > m
        return X_b + np.linalg.inv(np.linalg.inv(B) + H.T @ np.linalg.inv(R) @ H) @ (
            Y - H @ X_b
        )

    dims_Xb = (5, 3)
    dims_B = (5, 5)  # wrong in paper
    dims_H = (5, 5)
    dims_R = (5, 5)
    dims_Y = (5, 3)
    dims_out = dims_Xb

    random_inputs_invariants(EKF, dims_out, dims_Xb, dims_B, dims_H, dims_R, dims_Y)
    # random_inputs_invariants(EKF, (8, 6), (8, 6), (8, 8), (8, 8), (8, 8), (8, 6))


def test_Image_Restoration(debug=False):
    def IR(H, y, v, u, scale_mat, scale_vec):
        # H: m x n, y: m x 1, v: n x 1, u: n x 1, n > m
        return np.linalg.inv(H.T @ H + scale_mat) @ (H.T @ y + scale_vec * (v - u))

    dims_H = (3, 5)
    dims_y = (3, 1)
    dims_v = (5, 1)
    dims_u = (5, 1)
    dims_mat = (5, 5)
    dims_vec = (5, 1)
    dims_out = (5, 1)

    random_inputs_invariants(
        IR, dims_out, dims_H, dims_y, dims_v, dims_u, dims_mat, dims_vec, debug=debug
    )


def test_Randomized_Matrix_Inversion(debug=False):
    def RMI_1(S, A, W, X):
        L = S @ np.linalg.inv(S.T @ A.T @ W @ A @ S) @ S.T
        I = np.eye(X.shape[0])
        return X + (I - X @ A.T) @ L @ A.T @ W

    def RMI_2(S, A, X):
        SAS_inv = S @ np.linalg.inv(S.T @ A @ S) @ S.T
        I = np.eye(X.shape[0])
        return SAS_inv + (I - SAS_inv) @ X @ (I - A @ SAS_inv)

    n = 5
    q = 3
    dims_S = (n, q)
    dims_A = (n, n)
    dims_W = (n, n)
    dims_X = (n, n)
    dims_out = dims_X

    random_inputs_invariants(RMI_1, dims_out, dims_S, dims_A, dims_W, dims_X, debug=debug)
    random_inputs_invariants(RMI_2, dims_out, dims_S, dims_A, dims_X, debug=debug)


def test_Stochastic_Newton(debug=False):
    # we do not differentiate by `k`
    def SN(B, A, W):
        I_l = np.eye(W.shape[1])
        inverse = np.linalg.inv(I_l + W.T @ A @ B @ A.T @ W)
        I_n = np.eye(B.shape[0])
        return B @ (I_n - A.T @ W @ inverse @ W.T @ A @ B)

    l = 2
    m = 3
    n = 5
    dims_B = (n, n)
    dims_A = (m, n)
    dims_W = (m, l)
    dims_out = dims_B

    random_inputs_invariants(SN, dims_out, dims_B, dims_A, dims_W, debug=debug)


def test_Tikhonov_Regularization(debug=False):
    def TR(A, G, b):
        return np.linalg.inv(A.T @ A + G.T @ G) @ A.T @ b

    def GTR(A, P, Q, b, x0):
        return np.linalg.inv(A.T @ P @ A + Q) @ (A.T @ P @ b + Q @ x0)

    n = 5
    m = 3
    dims_A = (n, m)
    dims_G = (m, m)
    dims_b = (n, 1)
    dims_out = (m, 1)

    random_inputs_invariants(TR, dims_out, dims_A, dims_G, dims_b, debug=debug)

    dims_P = (n, n)
    dims_Q = (m, m)
    dims_x0 = (m, 1)

    random_inputs_invariants(GTR, dims_out, dims_A, dims_P, dims_Q, dims_b, dims_x0)


def test_LMMSE(debug=False):
    def LMMSE(CX, A, CZ, y, x):
        return CX @ A.T @ np.linalg.inv(A @ CX @ A.T + CZ) @ (y - A @ x) + x

    n = 5
    m = 3
    dims_CX = (n, n)
    dims_A = (m, n)
    dims_CZ = (m, m)
    dims_y = (m, 1)
    dims_x = (n, 1)
    dims_out = dims_x

    random_inputs_invariants(
        LMMSE, dims_out, dims_CX, dims_A, dims_CZ, dims_y, dims_x, debug=debug
    )


def test_Kalman_Filter(debug=False):
    # TODO: this would be a nice test case for multiple returns (P_k, x_k) and differentiable subroutines (K_k)
    def KF_P(P, H, R):
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        I = np.eye(K.shape[0])
        return (I - K @ H) @ P

    def KF_x(x, K, z, H):
        return x + K @ (z - H @ x)

    n = 4
    m = 6
    dims_P = (n, n)
    dims_H = (m, n)
    dims_R = (m, m)
    dims_x = (n, 1)
    dims_K = (n, m)
    dims_z = (m, 1)

    random_inputs_invariants(KF_P, dims_P, dims_P, dims_H, dims_R, debug=debug)
    random_inputs_invariants(KF_x, dims_x, dims_x, dims_K, dims_z, dims_H, debug=debug)


if __name__ == "__main__":
    test_central_fd()
    test_mma()
    test_elementwise()
    test_quadric()
    test_inverse()
    test_gls()
    test_Optimization()
    test_Signal_Processing()
    test_Ensemble_Kalman()
    test_Image_Restoration()
    test_Randomized_Matrix_Inversion()
    test_Stochastic_Newton()
    test_Tikhonov_Regularization()
    test_LMMSE()
    test_Kalman_Filter()
