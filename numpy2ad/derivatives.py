import ast
from enum import Enum
from copy import deepcopy

# TODO:
# replace deepcopy (slow) with some method to generate ast.Call
# use linalg.solve for inverse


class WithRespectTo(Enum):
    Left = 0
    Right = 1


def derivative_UnOp(node: ast.UnaryOp) -> ast.Expr:
    match type(node.op):
        case ast.USub:  # b = -a
            return ast.Expr(value=ast.Constant(value=-1.0))
        case _:
            raise ValueError("Not implemented yet.")


def derivative_BinOp(node: ast.BinOp, wrt: WithRespectTo) -> ast.Expr:
    """
    Returns an expression that contains the partial derivative
    of the given binary operation with respect to the specified argument (Left or Right).
    Only for scalar variables (and operators that are overloaded for ndarrays)
    """
    match type(node.op):
        case ast.Add:  # c = l + r
            return ast.Expr(value=ast.Constant(value=1.0))
        case ast.Sub:  # c = l - r
            return ast.Expr(
                value=ast.Constant(value=(1.0 if wrt is WithRespectTo.Left else -1.0))
            )
        case ast.Mult:  # c = l * r
            return ast.Expr(value=(node.right if wrt is WithRespectTo.Left else node.left))
        case ast.Div:  # c = a / b -> 1/b or -a / b**2
            return ast.Expr(
                value=(
                    ast.BinOp(op=ast.Div(), left=ast.Constant(value=1.0), right=node.right)
                    if wrt is WithRespectTo.Left
                    else ast.BinOp(
                        op=ast.Div(),
                        left=ast.UnaryOp(op=ast.USub(), operand=node.left),
                        right=ast.BinOp(
                            op=ast.Pow(), left=node.right, right=ast.Constant(value=2)
                        ),
                    )
                )
            )
        case ast.MatMult:  # C = A @ B -> A_a = C_a B^T, B_a = A^T C_a (order is handled by calling method)
            transpose = ast.Attribute(
                value=ast.Name(id="", ctx=ast.Load()), attr="T", ctx=ast.Load()
            )
            if wrt is WithRespectTo.Left:
                transpose.value.id = node.right.id
            else:
                transpose.value.id = node.left.id
            return ast.Expr(value=transpose)
        case ast.Pow:  # b = a**c -> b' = c a**(c-1). c must be const!
            new_right = ast.BinOp(
                op=ast.Pow(),
                left=node.left,
                right=ast.BinOp(op=ast.Sub(), left=node.right, right=ast.Constant(value=1)),
            )

            value = ast.BinOp(op=ast.Mult(), left=node.right, right=new_right)
            return ast.Expr(value=value) if wrt is WithRespectTo.Left else None

        case _:
            raise TypeError("Not implemented yet")


def derivative_Call(call: ast.Call, wrt_arg: int) -> ast.Expr:
    func = call.func.attr  # e.g. "exp"

    # def _make_numpy_Call(type: str, args: list):
    #     # required: args and keywords
    #     function = ast.Attribute(value="numpy", attr=type, ctx=ast.Load())
    #     return ast.Call(func=function, args=args, keywords=[])

    match func:
        case "inv":  # numpy.linalg.inv
            assert call.func.value.attr == "linalg"
            # B = A^-1 -> A_a += -A^-T @ B_a @ A^-T
            A_inv = ast.Name(id="A_inv", ctx=ast.Load())
            A_inv_T = ast.Attribute(attr="T", ctx=ast.Load(), value=A_inv)
            B_a = ast.Name(id="B_a", ctx=ast.Load())
            binop_right = ast.BinOp(op=ast.MatMult(), left=B_a, right=A_inv_T)
            return ast.Expr(
                value=ast.UnaryOp(
                    op=ast.USub(),
                    operand=ast.BinOp(op=ast.MatMult(), left=A_inv_T, right=binop_right),
                )
            )
        case "exp":
            return ast.Expr(value=call)
        case "divide":  # a / b
            if wrt_arg == 0:  # 1 / b
                recip = deepcopy(call.func)
                recip.attr = "reciprocal"  # does not work for integers!

                return ast.Expr(value=ast.Call(func=recip, args=[call.args[1]], keywords=[]))

            else:  # - a / b ** 2
                minus_a = ast.UnaryOp(op=ast.USub(), operand=call.args[0])
                b_squared = deepcopy(call)
                b_squared.func.attr = "square"
                b_squared.args = [call.args[1]]
                divide = deepcopy(call.func)
                divide.attr = "divide"

                return ast.Expr(
                    value=ast.Call(func=divide, args=[minus_a, b_squared], keywords=[])
                )
        case "square":  # x^2
            mult = deepcopy(call.func)
            mult.attr = "multiply"
            return ast.Expr(
                value=ast.Call(
                    func=mult, args=[call.args[0], ast.Constant(value=2.0)], keywords=[]
                )
            )
        case "ones":
            return None
        case "zeros":
            return None
        case "transpose":
            return None
        case "full":
            return None
        case "diag":
            return None
        case "eye":
            return None
        case _:
            raise ValueError("Not implemented yet")
