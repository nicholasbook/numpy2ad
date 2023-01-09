import numpy as np

expr = "D = A @ B + C"
# goal:
# v0 = A @ B
# D = v0 + C
# v0_a = zeros(v0.shape)
# C_a += D_a
# v0_a += D_a
# B_a += A.T @ v0_a
# A_a += v0_a @ B.T


def simple_exp(A, B, C):
    return A @ B + C


def simple_exp_SAC(A, B, C):
    v0 = A
    v1 = B
    v2 = C
    v3 = v0 @ v1
    v4 = v3 + v2
    return v4


def simple_exp_ad(A, B, C, A_a, B_a, C_a):
    # forward SAC (no overwriting)
    v0 = A
    v1 = B
    v2 = C

    v3 = v0 @ v1
    v4 = v3 + v2

    # initialize argument adjoints §1 AR
    v2_a = C_a
    v1_a = B_a
    v0_a = A_a

    # initialize intermediate adjoints
    v3_a = 0

    # initialize adjoint direction ("y_a")
    v4_a = 1  # np.ones ?

    # reverse mode (no values restored) §2 AR
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

    AB = np.dot(A, B)
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
    H = -np.ones(G.shape)  # v10 = -1
    return (G - H) * (G + H)  # v11 = v9 - v10; v12=v9+v10; v13=v11*v12;
    # return v13


def quadratic(A, x, b):
    return np.transpose(x) @ A @ x - np.transpose(x) @ b


def sigmoid(A: np.ndarray):
    denominator = np.ones(A.shape) + np.exp(-A)
    return np.divide(np.ones(A.shape), denominator)


def OLS(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Ordinary Least Squares

    Args:
        X (np.ndarray): design matrix
        y (np.ndarra<): observations vector

    Returns:
        np.ndarray: optimal coefficient vector
    """
    return np.linalg.inv(X.T @ X) @ X.T @ y


def GLS(X: np.ndarray, M: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Generalized Least Squares
    Model: y = Xb + e, E[e|X] = 0, Cov[e|X] = M

    Args:
        X (np.ndarray): design matrix (predictor values)
        M (np.ndarray): covariance matrix (of error e given X)
        y (np.ndarray): oberservations vector

    Returns:
        np.ndarray: optimal coefficient vector
    """
    M_inv = np.linalg.inv(M)
    return np.linalg.inv(X.T @ M_inv @ X) @ X.T @ M_inv @ y
