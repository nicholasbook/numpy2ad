import numpy as np
from numpy2ad import transform
import timeit
from numba import njit


def MMA(A, B, C):
    return A @ B + C


@njit
def MMA_jit(A, B, C):
    tmp = A @ B
    return tmp + C


@njit
def MMA_ad_jit(A, B, C, A_a, B_a, C_a, out_a):
    v0 = A @ B
    out = v0 + C
    v0_a = np.zeros_like(v0)
    C_a += out_a
    v0_a += out_a
    B_a += A.T @ v0_a
    A_a += v0_a @ B.T
    return (out, A_a, B_a, C_a)


def benchmark_mma(rows, func):
    # random square matrices
    A = np.random.rand(rows, rows)
    B = np.random.rand(rows, rows)
    C = np.random.rand(rows, rows)
    _ = func(A, B, C)


def benchmark_mma_ad(rows, func_ad):
    # initialize
    A = np.random.rand(rows, rows)
    B = np.random.rand(rows, rows)
    C = np.random.rand(rows, rows)
    A_a = np.zeros_like(A)
    B_a = np.zeros_like(B)
    C_a = np.zeros_like(C)
    out_a = np.zeros_like(C)
    out_a[0, 0] = 1.0

    _, _, _, _ = func_ad(A, B, C, A_a, B_a, C_a, out_a)


def warm_up_the_jit():
    rows = 64
    A = np.random.rand(rows, rows)
    B = np.random.rand(rows, rows)
    C = np.random.rand(rows, rows)
    A_a = np.zeros_like(A)
    B_a = np.zeros_like(B)
    C_a = np.zeros_like(C)
    out_a = np.zeros_like(C)

    _ = MMA_jit(A, B, C)
    _, _, _, _ = MMA_ad_jit(A, B, C, A_a, B_a, C_a, out_a)


if __name__ == "__main__":
    num_rows = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    export_array = np.zeros(shape=(len(num_rows), 5))

    # generate MMA_ad
    exec(compile(transform(MMA), filename="<ast>", mode="exec"))

    warm_up_the_jit()

    average_over = 10
    for i, row in enumerate(num_rows):
        mma_result = timeit.timeit(
            "benchmark_mma(row, MMA)",
            setup="from __main__ import benchmark_mma, MMA",
            globals=locals(),
            number=average_over,
        )
        print(f"Forward pass with {row} rows took {mma_result} seconds.")

        # with Numba
        mma_jit_result = timeit.timeit(
            "benchmark_mma(row, MMA_jit)",
            setup="from __main__ import benchmark_mma, MMA_jit",
            globals=locals(),
            number=average_over,
        )
        print(
            f"[Numba JIT] Forward pass with {row} rows took {mma_jit_result} seconds."
        )

        # transformed code
        mma_ad_result = timeit.timeit(
            "benchmark_mma_ad(row, MMA_ad)",
            setup="from __main__ import benchmark_mma_ad",
            globals=locals(),
            number=average_over,
        )
        print(f"Forward and reverse pass with {row} rows took {mma_ad_result} seconds.")

        mma_ad_jit_result = timeit.timeit(
            "benchmark_mma_ad(row, MMA_ad_jit)",
            setup="from __main__ import benchmark_mma_ad, MMA_ad_jit",
            globals=locals(),
            number=average_over,
        )
        print(
            f"[Numba JIT] Forward and reverse pass with {row} rows took {mma_ad_jit_result} seconds.\n"
        )
        export_array[i, :] = [
            row,
            mma_result / average_over,
            mma_jit_result / average_over,
            mma_ad_result / average_over,
            mma_ad_jit_result / average_over,
        ]

    np.savetxt("timeit_mma.txt", export_array)

    # TODO: plots
