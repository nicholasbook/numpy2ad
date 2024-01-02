import numpy as np
from numpy2ad import transform
import timeit
import argparse


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
    out_a = np.zeros((X.shape[1], 1))
    out_a[0] = 1.0

    _, _, _, _ = func_ad(X, M, y, X_a, M_a, y_a, out_a)


def benchmark_gls_ad_full(X, M, y, func_ad):
    # initialize
    X_a = np.zeros_like(X)
    M_a = np.zeros_like(M)
    y_a = np.zeros_like(y)
    out_a = np.zeros((X.shape[1], 1))

    for index in range(out_a.size):  # loop over all output adjoints
        out_a.flat[index] = 1.0
        _, X_a, M_a, y_a = func_ad(X, M, y, X_a, M_a, y_a, out_a)

        # reset
        out_a.flat[index] = 0.0
        X_a = np.zeros_like(X)
        M_a = np.zeros_like(M)
        y_a = np.zeros_like(y)


def central_fd(func, *args: np.ndarray, wrt: int, index: int) -> float:
    # inputs_copy = [i.copy() for i in list(args)]
    inputs = list(args)

    def round_to_binary(h):
        return np.power(2.0, np.round(np.log(h) / np.log(2.0)))

    value = (inputs[wrt]).flat[index].copy()
    h = round_to_binary(np.cbrt(np.finfo(np.float64).eps) * (1.0 + np.abs(value)))  # ~7.63e-06

    (inputs[wrt]).flat[index] = value - h
    y0 = func(*inputs)

    (inputs[wrt]).flat[index] = value + h
    y1 = func(*inputs)

    (inputs[wrt]).flat[index] = value

    return (y1 - y0) / (2.0 * h)


def benchmark_gls_cfd(X, M, y):
    # perturb all entries in X, M, and y
    for wrt, x in enumerate([X, M, y]):
        for index in range(x.size):
            dydx = central_fd(GLS, X, M, y, wrt=wrt, index=index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")  # runs each benchmark only once
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--target_time", type=float)  # in seconds (ignored when --once is set)
    args = parser.parse_args()

    # generate GLS_ad
    exec(compile(transform(GLS), filename="<ast>", mode="exec"))

    num_measures = 25

    # we want every run to take approx the same time in total, e.g. one minute
    # it scales as O(n^3) with some unknown constant
    num_rows_fwd_rev = (
        np.logspace(start=6.0, stop=13.0, num=num_measures, endpoint=True, base=2.0)
        .round()
        .astype(int)
    )  # 2**6 = 64, 2**13 = 8192

    ###### adjust this for your machine ######
    # set avg_over to 1 and adjust constants with worst s/iter
    goal_time = args.target_time if args.target_time is not None else 10.0  # seconds

    start = goal_time / (0.0002)  # N=64
    stop = goal_time / (9.4)  # N=8192

    ##########################################

    avg_over = (
        np.ones_like(num_rows_fwd_rev, dtype=int)
        if args.once
        else np.logspace(
            start=np.log2(start),
            stop=np.log2(stop),
            num=num_measures,
            endpoint=True,
            base=2.0,
        )
    )
    factor = 1.0  # optional

    results_fwd_rev = np.zeros(
        shape=(len(num_rows_fwd_rev), 6)
    )  # rows | fwd | fwd & rev | rel cost | fd | rel cost fd

    for i, rows in enumerate(num_rows_fwd_rev):
        average_over = int(factor * avg_over[i])

        print(f"Running for {average_over=} ...")

        # initialize
        X = np.random.rand(rows, rows // 8)  # normally way less columns
        M = np.random.rand(rows, rows)
        row_sum = np.sum(np.abs(M), axis=1)
        np.fill_diagonal(M, row_sum)
        M /= np.max(M)
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
        print(f"GLS with {rows=} took {gls_result}s ({gls_result / average_over} s/iter).")

        # transformed code
        # warm up
        for _ in range(average_over // 10):
            benchmark_gls_ad(X, M, y, GLS_ad)

        # repeated = []
        # for _ in range(5):
        gls_ad_result = timeit.timeit(
            "benchmark_gls_ad(X, M, y, GLS_ad)",
            setup="from __main__ import benchmark_gls_ad",
            globals=locals(),
            number=average_over,
        )
            # repeated.append(gls_ad_result)
        # print(repeated)
        print(
            f"Adjoint with {rows=} took {gls_ad_result}s ({gls_ad_result / average_over} s/iter)."
        )

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
        print(f"FD with {rows=} took {gls_fd_result} ({gls_fd_result / average_over} s/iter).")

        results_fwd_rev[i, :] = [
            rows,
            gls_result / average_over,  # avg. forward pass
            gls_ad_result / average_over,  # avg. adjoint
            gls_ad_result / gls_result,  # rel. cost adj/fd
            gls_fd_result / average_over,  # fd
            gls_fd_result / gls_result,  # rel. cost fd
        ]
        print("")

    np.savetxt("results/timeit_gls_fwd_rev.txt", results_fwd_rev)

    # --------- full Jacobian ----------

    if args.full:
        num_rows_full = (
            np.logspace(start=3.0, stop=7.0, num=num_measures, endpoint=True, base=2.0)
            .round()
            .astype(int)
        )  # 2**3 = 8, 2**7 = 128

        ###### adjust this for your machine ######
        # set avg_over to 1 and adjust constants with worst s/iter
        start = goal_time / (0.006)  # N=8
        stop = goal_time / (15.0)  # N=

        avg_over = (
            np.ones_like(num_rows_full, dtype=int)
            if args.once
            else np.logspace(
                start=np.log10(start).round(),
                stop=np.log10(stop).round(),
                num=num_measures,
                endpoint=True,
                base=10.0,
            )
        )
        factor = 1.0  # optional

        results_full = np.zeros(shape=(len(num_rows_full), 4))  # rows | adjoint | cfd | rel cost

        for i, rows in enumerate(num_rows_full):
            average_over = int(factor * avg_over[i])

            X = np.random.rand(rows, rows // 8)
            M = np.random.rand(rows, rows)
            row_sum = np.sum(np.abs(M), axis=1)
            np.fill_diagonal(M, row_sum)
            M /= np.max(M)
            y = np.random.rand(rows, 1)

            print(f"Running full Jacobian for {average_over=}...")

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
            print(
                f"Adjoint full Jacobian with {rows=} took {gls_ad_full_result}s ({gls_ad_full_result / average_over} s/iter)."
            )

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
            print(
                f"CFD full Jacobian with {rows=} took {gls_cfd_result}s ({gls_cfd_result / average_over} s/iter)."
            )

            print("")

            results_full[i, :] = [
                rows,
                gls_ad_full_result / average_over,  # avg. full jacobian
                gls_cfd_result / average_over,  # avg. full tangent jacobian
                gls_cfd_result / gls_ad_full_result,  # relative cost of tangent approximation
            ]

        np.savetxt("results/timeit_gls_full.txt", results_full)
