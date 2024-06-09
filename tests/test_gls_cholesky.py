import numpy as np
import scipy
import scipy.linalg


def GLS(X, M, y):
    M_inv = np.linalg.inv(M)
    return np.linalg.inv(X.T @ M_inv @ X) @ X.T @ M_inv @ y


def GLS_cholesky(X, M, y):
    L0 = np.linalg.cholesky(M)
    v0 = scipy.linalg.solve_triangular(a=L0, b=X, trans="T", lower=True)  # v0.T = X^T L^-1
    v1 = v0.T  # "\tilde{X}"
    v2 = v1 @ v0  # X^T L^-1 (X^T L^-1)^T
    L1 = np.linalg.cholesky(v2)  # "\tilde{L}"
    v3 = scipy.linalg.solve_triangular(a=L1, b=v1, lower=True)  # "z"
    v4 = scipy.linalg.solve_triangular(a=L1, b=v3, trans="T", lower=True)  # "Z"
    v5 = v4.T
    v6 = scipy.linalg.solve_triangular(a=L0, b=v5, lower=True)
    v7 = v6.T  # "\tilde{Z}"
    return v7 @ y


def GLS_ad(X, M, y, X_a, M_a, y_a, out_a):
    M_inv = np.linalg.inv(M)
    v0 = X.T
    v1 = v0 @ M_inv
    v2 = v1 @ X
    v3 = np.linalg.inv(v2)
    v4 = v3 @ v0
    v5 = v4 @ M_inv
    out = v5 @ y

    M_inv_a = np.zeros_like(M_inv)
    v0_a = X_a.T
    v1_a = np.zeros_like(v1)
    v2_a = np.zeros_like(v2)
    v3_a = np.zeros_like(v3)
    v4_a = np.zeros_like(v4)
    v5_a = np.zeros_like(v5)

    y_a += v5.T @ out_a
    v5_a += out_a @ y.T
    M_inv_a += v4.T @ v5_a
    v4_a += v5_a @ M_inv.T
    v0_a += v3.T @ v4_a
    v3_a += v4_a @ v0.T
    v2_a -= v3.T @ (v3_a @ v3.T)
    X_a += v1.T @ v2_a
    v1_a += v2_a @ X.T
    M_inv_a += v0.T @ v1_a
    v0_a += v1_a @ M_inv.T
    M_a -= M_inv.T @ (M_inv_a @ M_inv.T)
    return (out, X_a, M_a, y_a)


def GLS_cholesky_ad(X, M, y, X_a, M_a, y_a, out_a):
    L0 = np.linalg.cholesky(M)  # M = L L^T
    v0 = scipy.linalg.solve_triangular(
        a=L0, b=X, trans="T", lower=True
    )  # v0.T = X^T L^-1 <-> L^T v0 = X
    v1 = v0.T  # "\tilde{X}"
    v2 = v1 @ v0  # X^T L^-1 (X^T L^-1)^T
    L1 = np.linalg.cholesky(v2)  # "\tilde{L}"
    v3 = scipy.linalg.solve_triangular(a=L1, b=v1, lower=True)  # "z"
    v4 = scipy.linalg.solve_triangular(a=L1, b=v3, trans="T", lower=True)  # "Z"
    v5 = v4.T
    v6 = scipy.linalg.solve_triangular(a=L0, b=v5, lower=True)  # L \tilde{Z}^T = Z^T
    v7 = v6.T  # "\tilde{Z}"
    out = v7 @ y

    # reverse mode
    y_a += v7.T @ out_a
    v7_a = out_a @ y.T
    v6_a = v7_a.T

    v5_a = scipy.linalg.solve_triangular(
        a=L0, b=v6_a, trans="T", lower=True
    )  # L^T Z_a^T = \tilde{Z_a}
    L0_a = -v5_a @ v6.T
    v4_a = v5_a.T

    v3_a = scipy.linalg.solve_triangular(a=L1, b=v4_a, lower=True)  # \tilde{L} z_a = Z_a
    # L1_a = -v4 @ v3_a.T # not needed?

    v1_a = scipy.linalg.solve_triangular(
        a=L1, b=v3_a, trans="T", lower=True
    )  # \tilde{L}^T \tilde{X} = z_a
    # L1_a -= v1_a @ v3.T

    v2_a = -v1_a @ v5  # (\tilde{L} \tilde{L}^T)_a = -\tilde{X}_a Z^T

    v1_a += v2_a @ v0.T
    # v0_a = v1.T @ v2_a
    v0_a = v1_a.T  # ? overwritten

    X_a += scipy.linalg.solve_triangular(a=L0, b=v0_a, lower=True)
    L0_a -= v0 @ X_a.T

    # Cholesky adjoint
    L0_T_L0_a = L0.T @ L0_a
    L0_a_lower = np.tril(L0_T_L0_a, k=-1) + 0.5 * np.diag(L0_T_L0_a)
    L0_a_x = scipy.linalg.solve_triangular(a=L0, b=L0_a_lower, trans="T", lower=True)
    M_a_T = scipy.linalg.solve_triangular(a=L0, b=L0_a_x.T, trans="T", lower=True)
    M_a = M_a_T.T

    return (out, X_a, M_a, y_a)


def test_cholesky_equal():
    N = 3

    M = np.random.rand(N, N)
    M = 0.5 * (M + M.T) + N * np.eye(N)
    # print(f"det(M) = {np.linalg.det(M)}")

    y = np.random.rand(N, 1)
    X = np.random.rand(N, N)

    # print(f"GLS(X, M, y) = {GLS(X, M, y)}")
    # print(f"GLS_cholesky(X, M, y) = {GLS_cholesky(X, M, y)}")

    # test forward mode

    assert np.allclose(GLS(X, M, y), GLS_cholesky(X, M, y))

    # test reverse mode

    X_a = np.zeros_like(X)
    M_a = np.zeros_like(M)
    y_a = np.zeros_like(y)
    out_a = np.zeros((N, 1))
    out_a[0, 0] = 1.0  # seed

    _, X_a, M_a, y_a = GLS_ad(X, M, y, X_a, M_a, y_a, out_a)

    X_a_c = np.zeros_like(X)
    M_a_c = np.zeros_like(M)
    y_a_c = np.zeros_like(y)

    _, X_a_c, M_a_c, y_a_c = GLS_cholesky_ad(X, M, y, X_a_c, M_a_c, y_a_c, out_a)

    assert np.allclose(X_a, X_a_c)
    assert np.allclose(M_a, M_a_c)
    assert np.allclose(y_a, y_a_c)


if __name__ == "__main__":
    test_cholesky_equal()
