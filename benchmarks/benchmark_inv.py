import numpy as np
from numpy2ad import transform
import timeit
import argparse


def matmul(A, B):
    return A @ B

def inverse(A):
    return np.linalg.inv(A)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")  # runs each benchmark only once
    parser.add_argument("--target_time", type=float)  # in seconds (ignored when --once is set)
    args = parser.parse_args()

    num_measures = 25

    # we want every run to take approx the same time in total, e.g. one minute
    # it scales as O(n^3) with some unknown constant
    N = (
        np.logspace(start=6.0, stop=13.0, num=num_measures, endpoint=True, base=2.0)
        .round()
        .astype(int)
    )  # 2**6 = 64, 2**13 = 8192

    ###### adjust this for your machine ######
    # set avg_over to 1 and adjust constants with worst s/iter
    goal_time = args.target_time if args.target_time is not None else 10.0  # seconds

    start = goal_time / (0.00001)  # N=64
    stop = goal_time / (3.2)  # N=8192

    ##########################################

    avg_over = (
        np.ones_like(N, dtype=int)
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

    results = np.zeros(
        shape=(len(N), 4)
    )  # rows | @ | inv | rel

    for i, rows in enumerate(N):
        average_over = int(factor * avg_over[i])

        print(f"Running for {average_over=} ...")

        # initialize
        A = np.random.rand(rows, rows)
        np.fill_diagonal(A, np.sum(np.abs(A), axis=1))
        B = np.random.rand(rows, rows)

        # matmul
        # warm up
        for _ in range(average_over // 10):
            _ = matmul(A, B)

        matmul_result = timeit.timeit(
            "matmul(A, B)",
            setup="from __main__ import matmul",
            globals=locals(),
            number=average_over,
        )
        print(f"Matmul with {rows=} took {matmul_result}s ({matmul_result / average_over} s/iter).")

        # inverse 
        # warm up
        for _ in range(average_over // 10):
            _ = inverse(A)

        # repeated = []
        # for _ in range(5):
        inverse_result = timeit.timeit(
            "inverse(A)",
            setup="from __main__ import inverse",
            globals=locals(),
            number=average_over,
        )
            # repeated.append(gls_ad_result)
        # print(repeated)
        print(
            f"Inverse with {rows=} took {inverse_result}s ({inverse_result / average_over} s/iter)."
        )


        results[i, :] = [
            rows,
            matmul_result / average_over,  # avg. matmul
            inverse_result / average_over,  # avg. inverse
            inverse_result / matmul_result,  # rel. cost matmul / inverse
        ]  
        print("")

    np.savetxt("results/timeit_inverse.txt", results)