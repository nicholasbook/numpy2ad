import numpy as np
from numpy2ad import transform
import timeit


def GLS(X, M, y):
    M_inv = np.linalg.inv(M)
    return np.linalg.inv(X.T @ M_inv @ X) @ X.T @ M_inv @ y


def benchmark_gls(X, M, y):
    _ = GLS(X, M, y)


def benchmark_gls_ad(X, M, y, func_ad):
    # initialize
    X_a = np.zeros_like(X)
    M_a = np.zeros_like(M)
    y_a = np.zeros_like(y)
    out_a = np.zeros_like(y)
    out_a[0, 0] = 1.0

    _, _, _, _ = func_ad(X, M, y, X_a, M_a, y_a, out_a)


def benchmark_gls_ad_full(X, M, y, func_ad):
    # initialize
    X_a = np.zeros_like(X)
    M_a = np.zeros_like(M)
    y_a = np.zeros_like(y)
    out_a = np.zeros_like(y)

    for index in range(out_a.size):  # loop over all output adjoints
        out_a.flat[index] = 1.0
        _, X_a, M_a, y_a = func_ad(X, M, y, X_a, M_a, y_a, out_a)

        # reset
        out_a.flat[index] = 0.0
        X_a = np.zeros_like(X)
        M_a = np.zeros_like(M)
        y_a = np.zeros_like(y)


def central_fd(func, *args: np.ndarray, wrt: int, index: int) -> float:
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


def benchmark_gls_cfd(X, M, y):
    # perturb all entries in X, M, and y
    for wrt, x in enumerate([X, M, y]):
        for index in range(x.size):
            dydx = central_fd(GLS, X, M, y, wrt=wrt, index=index)


if __name__ == "__main__":
    # generate GLS_ad
    exec(compile(transform(GLS), filename="<ast>", mode="exec"))

    num_rows_fwd_rev = [64, 128, 256, 512, 1024, 1536, 2048]  # , 3072, 4096]
    avg_over = [1e5, 1e5, 1e4, 1e3, 1e3, 1e2, 1e2]  # , 1e1, 1e1]
    factor = 0.1

    results_fwd_rev = np.zeros(
        shape=(len(num_rows_fwd_rev), 6)
    )  # rows | fwd | fwd & rev | rel cost | fd | rel cost fd

    for i, rows in enumerate(num_rows_fwd_rev):
        average_over = int(factor * avg_over[i])

        # initialize
        X = np.random.rand(rows, rows)
        M = np.random.rand(rows, rows)
        y = np.random.rand(rows, 1)

        # forward pass
        # warm up
        for _ in range(average_over // 10):
            benchmark_gls(X, M, y)

        gls_result = timeit.timeit(
            "benchmark_gls(X, M, y)",
            setup="from __main__ import benchmark_gls, GLS",
            globals=locals(),
            number=average_over,
        )
        print(f"GLS with {rows=} took {gls_result} seconds.")

        # transformed code
        # warm up
        for _ in range(average_over // 10):
            benchmark_gls_ad(X, M, y, GLS_ad)

        gls_ad_result = timeit.timeit(
            "benchmark_gls_ad(X, M, y, GLS_ad)",
            setup="from __main__ import benchmark_gls_ad",
            globals=locals(),
            number=average_over,
        )
        print(f"Adjoint with {rows=} took {gls_ad_result} seconds.")

        # finite difference

        # warm up
        for _ in range(average_over // 10):
            central_fd(GLS, X, M, y, wrt=0, index=0)

        gls_fd_result = timeit.timeit(
            "central_fd(GLS, X, M, y, wrt=0, index=0)",
            setup="from __main__ import GLS, central_fd",
            globals=locals(),
            number=average_over,
        )
        print(f"FD with {rows=} took {gls_fd_result} seconds.")

        results_fwd_rev[i, :] = [
            rows,
            gls_result / average_over,  # avg. forward pass
            gls_ad_result / average_over,  # avg. forward + reverse
            gls_ad_result / gls_result,  # rel cost
            gls_fd_result / average_over,  # fd
            gls_fd_result / gls_result,  # rel cost fd
        ]
        print("")

    np.savetxt("timeit_gls_fwd_rev.txt", results_fwd_rev)

    # --------- full Jacobian ----------

    num_rows_full = [8, 16, 32, 64]  # , 96]
    avg_over = [100, 100, 10, 10]  # 5]
    factor = 0.5

    results_full = np.zeros(shape=(len(num_rows_full), 4))  # rows | adjoint | cfd | rel cost

    for i, rows in enumerate(num_rows_full):
        average_over = int(factor * avg_over[i])

        X = np.random.rand(rows, rows)
        M = np.random.rand(rows, rows)
        y = np.random.rand(rows, 1)

        # adjoint full jacobian
        # warm up
        for _ in range(average_over // 10):
            benchmark_gls_ad_full(X, M, y, GLS_ad)

        gls_ad_full_result = timeit.timeit(
            "benchmark_gls_ad_full(X, M, y, GLS_ad)",
            setup="from __main__ import benchmark_gls_ad_full",
            globals=locals(),
            number=average_over,
        )
        print(f"Adjoint full Jacobian with {rows=} took {gls_ad_full_result} seconds.")

        # cfd full jacobian
        # warm up
        for _ in range(average_over // 10):
            benchmark_gls_cfd(X, M, y)

        gls_cfd_result = timeit.timeit(
            "benchmark_gls_cfd(X, M, y)",
            setup="from __main__ import benchmark_gls_cfd",
            globals=locals(),
            number=average_over,
        )
        print(f"CFD full Jacobian with {rows=} took {gls_cfd_result} seconds.")

        print("")

        results_full[i, :] = [
            rows,
            gls_ad_full_result / average_over,  # avg. full jacobian
            gls_cfd_result / average_over,  # avg. full tangent jacobian
            gls_cfd_result / gls_ad_full_result,  # relative cost of tangent approximation
        ]

    np.savetxt("timeit_gls_full.txt", results_full)
