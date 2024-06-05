import numpy as np
import scipy


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
