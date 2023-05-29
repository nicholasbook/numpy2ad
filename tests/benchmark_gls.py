import numpy as np
from numpy2ad import transform
import timeit
from numba import njit


def GLS(X, M, y):
    M_inv = np.linalg.inv(M)
    return np.linalg.inv(X.T @ M_inv @ X) @ X.T @ M_inv @ y


@njit
def GLS_jit(X, M, y):
    M_inv = np.linalg.inv(M)
    return np.linalg.inv(X.T @ M_inv @ X) @ X.T @ M_inv @ y


@njit
def GLS_ad_jit(X, M, y, X_a, M_a, y_a, out_a):
    M_inv = np.linalg.inv(M)
    v0 = X.T
    v1 = v0 @ M_inv
    v2 = v1 @ X
    v3 = np.linalg.inv(v2)
    v4 = X.T
    v5 = v3 @ v4
    v6 = v5 @ M_inv
    out = v6 @ y
    M_inv_a = np.zeros_like(M_inv)
    v0_a = X_a.T
    v1_a = np.zeros_like(v1)
    v2_a = np.zeros_like(v2)
    v3_a = np.zeros_like(v3)
    v4_a = X_a.T
    v5_a = np.zeros_like(v5)
    v6_a = np.zeros_like(v6)
    y_a += v6.T @ out_a
    v6_a += out_a @ y.T
    M_inv_a += v5.T @ v6_a
    v5_a += v6_a @ M_inv.T
    v4_a += v3.T @ v5_a
    v3_a += v5_a @ v4.T
    v2_a -= v3.T @ (v3_a @ v3.T)
    X_a += v1.T @ v2_a
    v1_a += v2_a @ X.T
    M_inv_a += v0.T @ v1_a
    v0_a += v1_a @ M_inv.T
    M_a -= M_inv.T @ (M_inv_a @ M_inv.T)
    return (out, X_a, M_a, y_a)


def benchmark_gls(rows, func):
    # random square matrices
    X = np.random.rand(rows, rows)
    M = np.random.rand(rows, rows)
    y = np.random.rand(rows, 1)
    _ = func(X, M, y)


def benchmark_gls_ad(rows, func_ad):
    # initialize
    X = np.random.rand(rows, rows)
    M = np.random.rand(rows, rows)
    y = np.random.rand(rows, 1)
    X_a = np.zeros_like(X)
    M_a = np.zeros_like(M)
    y_a = np.zeros_like(y)
    out_a = np.zeros_like(y)
    out_a[0, 0] = 1.0

    _, _, _, _ = func_ad(X, M, y, X_a, M_a, y_a, out_a)


def warm_up_the_jit():
    rows = 64
    X = np.random.rand(rows, rows)
    M = np.random.rand(rows, rows)
    y = np.random.rand(rows, 1)
    X_a = np.zeros_like(X)
    M_a = np.zeros_like(M)
    y_a = np.zeros_like(y)
    out_a = np.zeros_like(y)

    _ = GLS_jit(X, M, y)
    _, _, _, _ = GLS_ad_jit(X, M, y, X_a, M_a, y_a, out_a)


if __name__ == "__main__":
    num_rows = [32, 64, 128, 256, 512, 1024, 2048, 4096]  # , 8192] # ~ 6 min for 4096

    # generate GLS_ad
    exec(compile(transform(GLS), filename="<ast>", mode="exec"))

    warm_up_the_jit()

    for row in num_rows:
        gls_result = timeit.timeit(
            "benchmark_gls(row, GLS)",
            setup="from __main__ import benchmark_gls, GLS",
            globals=locals(),
            number=20,
        )
        print(f"Forward pass with {row} rows took {gls_result} seconds.")

        # with Numba
        gls_jit_result = timeit.timeit(
            "benchmark_gls(row, GLS_jit)",
            setup="from __main__ import benchmark_gls, GLS_jit",
            globals=locals(),
            number=20,
        )
        print(
            f"[Numba JIT] Forward pass with {row} rows took {gls_jit_result} seconds."
        )

        # transformed code
        gls_ad_result = timeit.timeit(
            "benchmark_gls_ad(row, GLS_ad)",
            setup="from __main__ import benchmark_gls_ad",
            globals=locals(),
            number=20,
        )
        print(f"Forward and reverse pass with {row} rows took {gls_ad_result} seconds.")

        gls_ad_jit_result = timeit.timeit(
            "benchmark_gls_ad(row, GLS_ad_jit)",
            setup="from __main__ import benchmark_gls_ad, GLS_ad_jit",
            globals=locals(),
            number=20,
        )
        print(
            f"[Numba JIT] Forward and reverse pass with {row} rows took {gls_ad_jit_result} seconds.\n"
        )

    # TODO: plots
