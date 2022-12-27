import numpy


def simple_exp(A, B, C):
    return A @ B + C


# simple_exp should first be transformed to simple_exp_SAC """


def simple_exp_SAC(A, B, C):
    v0 = A
    v1 = B
    v2 = C
    v3 = v0 @ v1
    v4 = v3 + v2
    return v4


# simple_exp_SAC should then be transformed to simple_exp_ad
def simple_exp_adjoint(A, B, C, A_a, B_a, C_a):
    # forward SAC (no overwriting)
    v0 = A
    v1 = B
    v2 = C
    v3 = v0 @ v1
    v4 = v3 + v2

    # initialize argument adjoints §1 AR
    v0_a = A_a
    v1_a = B_a
    v2_a = C_a

    # initialize intermediate adjoints
    v3_a = 0

    # initialize adjoint direction ("y_a")
    v4_a = 1  # np.ones ?

    # reverse mode §2 AR
    v3_a += v4_a
    v2_a += v4_a
    v4_a = 0  # §3 AR
    v0_a += v1 * v3_a  # transpose?
    v1_a += v0 * v3_a
    v3_a = 0

    # return function value and all adjoints
    return (v4, v0_a, v1_a, v2_a)


# a bit more complex
def simple_exp_dot(A, B, C):
    # assert A.ndim == 2 and B.ndim == 2 and C.ndim == 2
    # assert A.shape[1] == B.shape[0] and C.shape == (A.shape[0], B.shape[1])

    AB = numpy.dot(A, B)
    return AB + C


def nested_expr3(A, B, C, D):
    return ((A + B) + C) + D


def nested_expr4(A, B, C, D, E):
    return (((A + B) + C) + D) + E


def simple_return(A, B, C):
    D = (A + B) * C
    return D


def multiline_BinOps(A, B, C):
    # v0 = A, v1 = B, v2 = C
    D = A + B  # v3 = v0 + v1
    E = A - C  # v4 = v0 - v2
    F = D / E  # v5 = v3 / v4
    G = (A + E) * (B + F) * D  # v7=v0+v4; v8=v1+v5; v9=v7*v8
    H = -1  # v10 = -1
    return (G - H) * (G + H)  # v11 = v9 - v10; v12=v9+v10; v13=v11*v12;
    # return v13


def LLS(A, y):
    # v0 = A; v1 = y
    A_T = numpy.transpose(A)  # v2 = np.transpose(v0)
    A_T_A = A_T @ A  # v3 = v2 @ A
    return numpy.inverse(A_T_A) @ A_T @ y
    # v4 = np.inverse(v3); v5=v4@v2; v6=v5@v1; return v6


def quadratic(A, x, b):
    return 0.5 * numpy.transpose(x) @ A @ x - numpy.tranpose(x) @ b
