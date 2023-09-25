import numpy as np
from numpy2ad import transform
import timeit
import gc


def MMA(A, B, C):
    return A @ B + C


def benchmark_mma(A, B, C):
    _ = MMA(A, B, C)


def benchmark_mma_ad(A, B, C, func_ad):
    # initialize
    A_a = np.zeros_like(A)
    B_a = np.zeros_like(B)
    C_a = np.zeros_like(C)
    out_a = np.zeros_like(C)
    out_a[0, 0] = 1.0

    _, _, _, _ = func_ad(A, B, C, A_a, B_a, C_a, out_a)


def benchmark_mma_ad_full(A, B, C, func_ad):
    # initialize
    A_a = np.zeros_like(A)
    B_a = np.zeros_like(B)
    C_a = np.zeros_like(C)
    out_a = np.zeros_like(C)

    for index in range(out_a.size):  # loop over all output adjoints
        out_a.flat[index] = 1.0
        _, _, _, _ = func_ad(A, B, C, A_a, B_a, C_a, out_a)

        # reset
        out_a.flat[index] = 0.0
        A_a = np.zeros_like(A)
        B_a = np.zeros_like(B)
        C_a = np.zeros_like(C)


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


def benchmark_mma_cfd(A, B, C):
    for wrt, X in enumerate([A, B, C]):
        for index in range(X.size):
            _ = central_fd(MMA, A, B, C, wrt=wrt, index=index)


if __name__ == "__main__":
    # generate MMA_ad
    exec(compile(transform(MMA), filename="<ast>", mode="exec"))

    # Benchmark forward & reverse pass
    if True:
        num_rows_fwd_rev = [64, 128, 256, 512, 1024, 2048, 4096]
        avg_over = [1e6, 5e5, 1e4, 5e3, 5e2, 1e2, 1e1]
        factor = 0.5

        results_fwd_rev = np.zeros(
            shape=(len(num_rows_fwd_rev), 6)
        )  # rows | fwd | fwd & rev | rel cost | fd | rel cost fd

        for i, rows in enumerate(num_rows_fwd_rev):
            average_over = int(factor * avg_over[i])

            gc.disable()  # garbage collection

            # intialize
            A = np.random.rand(rows, rows)
            B = np.random.rand(rows, rows)
            C = np.random.rand(rows, rows)

            # forward pass
            # warm up
            for _ in range(average_over // 10):
                benchmark_mma(A, B, C)

            mma_result = timeit.timeit(
                "benchmark_mma(A, B, C)",
                setup="from __main__ import benchmark_mma, MMA",
                globals=locals(),
                number=average_over,
            )
            print(f"MMA with {rows=} took {mma_result} seconds.")

            # transformed code
            # warm up again
            for _ in range(average_over // 10):
                benchmark_mma_ad(A, B, C, MMA_ad)

            mma_ad_result = timeit.timeit(
                "benchmark_mma_ad(A, B, C, MMA_ad)",
                setup="from __main__ import benchmark_mma_ad",
                globals=locals(),
                number=average_over,
            )
            print(f"Adjoint with {rows=} took {mma_ad_result} seconds.")

            # finite difference
            # warm up
            for _ in range(average_over // 10):
                central_fd(MMA, A, B, C, wrt=0, index=0)

            mma_fd_result = timeit.timeit(
                "central_fd(MMA, A, B, C, wrt=0, index=0)",
                setup="from __main__ import MMA, central_fd",
                globals=locals(),
                number=average_over,
            )
            print(f"FD with {rows=} took {mma_fd_result} seconds.")

            results_fwd_rev[i, :] = [
                rows,
                mma_result / average_over,  # avg. forward pass
                mma_ad_result / average_over,  # avg. forward + reverse
                mma_ad_result / mma_result,  # rel cost
                mma_fd_result / average_over,  # finite difference
                mma_fd_result / mma_result,
            ]
            print("")

            gc.enable()

        np.savetxt("timeit_mma_fwd_rev.txt", results_fwd_rev)

    # ---------- Benchmark full Jacobian ---------- #

    if True:
        num_rows_full = [8, 16, 32, 64, 96, 128]
        avg_over = [1e4, 1e3, 5e2, 1e2, 1e1, 1e1]
        factor = 0.5

        results_full = np.zeros(
            shape=(len(num_rows_full), 4)
        )  # rows | adjoint | cfd | rel cost

        for i, rows in enumerate(num_rows_full):
            average_over = int(factor * avg_over[i])

            gc.disable()  # garbage collection

            A = np.random.rand(rows, rows)
            B = np.random.rand(rows, rows)
            C = np.random.rand(rows, rows)

            # full jacobian
            # warm up

            for _ in range(average_over // 10):
                benchmark_mma_ad_full(A, B, C, MMA_ad)

            mma_ad_full_result = timeit.timeit(
                "benchmark_mma_ad_full(A, B, C, MMA_ad)",
                setup="from __main__ import benchmark_mma_ad_full",
                globals=locals(),
                number=average_over,
            )
            print(f"Adjoint full Jacobian with {rows=} took {mma_ad_full_result} seconds.")

            # finite difference
            # warm up

            for _ in range(average_over // 10):
                benchmark_mma_cfd(A, B, C)

            mma_cfd_result = timeit.timeit(
                "benchmark_mma_cfd(A, B, C)",
                setup="from __main__ import MMA, benchmark_mma_cfd",
                globals=locals(),
                number=average_over,
            )
            print(f"CFD full Jacobian with {rows=} took {mma_cfd_result} seconds.")
            print("")

            results_full[i, :] = [
                rows,
                mma_ad_full_result / average_over,  # avg. full jacobian
                mma_cfd_result / average_over,  # avg. full tangent jacobian
                mma_cfd_result / mma_ad_full_result,  # relative cost of tangent approximation
            ]

            gc.enable()

        np.savetxt("timeit_mma_full.txt", results_full)
