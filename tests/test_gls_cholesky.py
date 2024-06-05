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
    L0 = np.linalg.cholesky(M)
    v0 = scipy.linalg.solve_triangular(a=L0, b=X, trans="T", lower=True)  # v0.T = X^T L^-1
    v1 = v0.T  # "\tilde{X}"
    v2 = v0 @ v1  # X^T L^-1 (X^T L^-1)^T
    L1 = np.linalg.cholesky(v2)  # "\tilde{L}"
    v3 = scipy.linalg.solve_triangular(a=L1, b=v1, lower=True)  # "z"
    v4 = scipy.linalg.solve_triangular(a=L1, b=v3, trans="T", lower=True)  # "Z"
    v5 = v4.T
    v6 = scipy.linalg.solve_triangular(a=L0, b=v5, lower=True) # L \tilde{Z}^T = Z^T
    v7 = v6.T  # "\tilde{Z}"
    out = v7 @ y

    y_a += v7.T @ out_a
    v7_a = out_a @ y.T
    v6_a = v7_a.T
    v5_a = scipy.linalg.solve_triangular(a=L0, b=v6_a, trans="T", lower=True) # L^T Z_a^T = \tilde{Z_a}
    v4_a = v5_a.T
    v3_a = scipy.linalg.solve_triangular(a=L1, b=v4_a, lower=True) # \tilde{L} z_a = Z_a
    v1_a = scipy.linalg.solve_triangular(a=L1, b=v3_a, trans="T", lower=True) # \tilde{L}^T \tilde{X} = z_a
    v2_a = -v1_a @ v5 # (\tilde{L} \tilde{L}^T) = -\tilde{X_a} Z^T
    v1_a += v2_a @ v0.T
    v0_a = v1.T @ v2_a
    v0_a = v1_a.T # ? overwritten
    X_a += scipy.linalg.solve_triangular(a=L0, b=v0_a, lower=True) 
    M_a -= ... # ? TODO

    return (out, X_a, M_a, y_a)


def test_GLS_equal():
    N = 20

    M = np.random.rand(N, N)
    M = 0.5 * (M + M.T) + N * np.eye(N)
    # print(f"det(M) = {np.linalg.det(M)}")

    y = np.random.rand(N)
    X = np.random.rand(N, N)

    assert np.allclose(GLS(X, M, y), GLS_cholesky(X, M, y))

    # print(f"GLS(X, M, y) = {GLS(X, M, y)}")
    # print(f"GLS_cholesky(X, M, y) = {GLS_cholesky(X, M, y)}")


if __name__ == "__main__":
    test_GLS_equal()
