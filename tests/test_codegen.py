import numpy as np
from inspect import getsource
from textwrap import dedent
from numpy2ad import transform, transform_expr
from typing import Callable


def _strip_import(src: str) -> str:
    str_list = src.split("\n")
    str_list.pop(0)
    str_list.pop(0)
    return "\n".join(str_list) + "\n"


def test_strip_import():
    f_ad_str = "import numpy as np\n\ndef f_ad(x, x_a):\n   return (x, x_a)"
    f_ad_str_stripped = "def f_ad(x, x_a):\n   return (x, x_a)\n"
    assert _strip_import(f_ad_str) == f_ad_str_stripped


def is_transformed_to(f: Callable, f_ad: Callable) -> bool:
    """Checks if transform(f) is transformed to (handcoded) f_ad."""
    f_str = dedent(getsource(f))
    f_str_ad = transform(f_str)
    f_str_ad_stripped = _strip_import(f_str_ad)

    # compare to handcoded codegen
    target_f_ad = dedent(getsource(f_ad))

    if not f_str_ad_stripped == target_f_ad:
        raise AssertionError(f"\n{f_str_ad_stripped=}\nis not equal to\n{target_f_ad=}")
    else:
        return True


def test_expression():
    expr = "C = A + B"
    transformed_expr = transform_expr(expr)
    target_expr = "C = A + B\nB_a += C_a\nA_a += C_a"
    assert transformed_expr == target_expr


def test_return():
    def f(x):
        return x

    def f_ad(x, x_a):
        return (x, x_a)

    assert is_transformed_to(f, f_ad)


def test_return_add():
    def f(A, B):
        return A + B

    def f_ad(A, B, A_a, B_a, out_a):
        out = A + B
        B_a += out_a
        A_a += out_a
        return (out, A_a, B_a)

    assert is_transformed_to(f, f_ad)


def test_return_call():
    def f(x):
        return np.square(x)

    def f_ad(x, x_a, out_a):
        out = np.square(x)
        x_a += np.multiply(x, 2.0) * out_a
        return (out, x_a)

    assert is_transformed_to(f, f_ad)


def test_return_inv():
    def f(A):
        return np.linalg.inv(A)

    def f_ad(A, A_a, out_a):
        out = np.linalg.inv(A)
        A_a -= out.T @ (out_a @ out.T)
        return (out, A_a)

    assert is_transformed_to(f, f_ad)


def test_return_add_call():
    def f(A):
        return A + np.linalg.inv(A)

    def f_ad(A, A_a, out_a):
        v0 = np.linalg.inv(A)
        out = A + v0
        v0_a = np.zeros_like(v0)
        v0_a += out_a
        A_a += out_a
        A_a -= v0.T @ (v0_a @ v0.T)
        return (out, A_a)

    assert is_transformed_to(f, f_ad)


def test_assign():
    def f(A, B):
        C = A + B
        return C

    def f_ad(A, B, A_a, B_a, C_a):
        C = A + B
        B_a += C_a
        A_a += C_a
        return (C, A_a, B_a)

    assert is_transformed_to(f, f_ad)


def test_return_nested_binop():
    def f(A, B):
        C = A + B
        return A @ (B @ C)

    def f_ad(A, B, A_a, B_a, out_a):
        C = A + B
        v0 = B @ C
        out = A @ v0
        C_a = np.zeros_like(C)
        v0_a = np.zeros_like(v0)
        v0_a += A.T @ out_a
        A_a += out_a @ v0.T
        C_a += B.T @ v0_a
        B_a += v0_a @ C.T
        B_a += C_a
        A_a += C_a
        return (out, A_a, B_a)

    assert is_transformed_to(f, f_ad)


def test_nested_call():
    def f(A):
        B = np.linalg.inv(np.linalg.inv(A + A))
        return A @ B.T

    def f_ad(A, A_a, out_a):
        v0 = A + A
        v1 = np.linalg.inv(v0)
        B = np.linalg.inv(v1)
        v2 = B.T
        out = A @ v2
        v0_a = np.zeros_like(v0)
        v1_a = np.zeros_like(v1)
        B_a = np.zeros_like(B)
        v2_a = B_a.T
        v2_a += A.T @ out_a
        A_a += out_a @ v2.T
        v1_a -= B.T @ (B_a @ B.T)
        v0_a -= v1.T @ (v1_a @ v1.T)
        A_a += v0_a
        A_a += v0_a
        return (out, A_a)

    assert is_transformed_to(f, f_ad)


def test_elementwise():
    def ew_mult(A, B):
        return A * B
    
    def ew_mult_ad(A, B, A_a, B_a, out_a):
        out = A * B
        B_a += A * out_a
        A_a += B * out_a
        return (out, A_a, B_a)
    
    assert is_transformed_to(ew_mult, ew_mult_ad)


# debugging
if __name__ == "__main__":
    test_strip_import()
    test_expression()
    test_return_add()
    test_return_call()
    test_return_inv()
    test_return_add_call()
    test_assign()
    test_return_nested_binop()
    test_nested_call()
    test_elementwise()
