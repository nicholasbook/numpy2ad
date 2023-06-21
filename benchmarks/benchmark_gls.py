import numpy as np
from numpy2ad import transform
import timeit


def GLS(X, M, y):
    M_inv = np.linalg.inv(M)
    return np.linalg.inv(X.T @ M_inv @ X) @ X.T @ M_inv @ y


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


if __name__ == "__main__":
    num_rows = [32, 64, 128, 256, 512, 1024, 2048, 4096]  # ~ 6 min for 4096
    export_array = np.zeros(shape=(len(num_rows), 5))

    # generate GLS_ad
    exec(compile(transform(GLS), filename="<ast>", mode="exec"))

    average_over = 10
    for i, row in enumerate(num_rows):
        gls_result = timeit.timeit(
            "benchmark_gls(row, GLS)",
            setup="from __main__ import benchmark_gls, GLS",
            globals=locals(),
            number=average_over,
        )
        print(f"Forward pass with {row} rows took {gls_result} seconds.")

        # transformed code
        gls_ad_result = timeit.timeit(
            "benchmark_gls_ad(row, GLS_ad)",
            setup="from __main__ import benchmark_gls_ad",
            globals=locals(),
            number=average_over,
        )
        print(f"Forward and reverse pass with {row} rows took {gls_ad_result} seconds.")

        export_array[i, :] = [
            row,
            gls_result / average_over,
            gls_ad_result / average_over
        ]

    np.savetxt("timeit_gls.txt", export_array)

