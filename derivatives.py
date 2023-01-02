import ast
from enum import Enum
from copy import deepcopy

# TODO: @/np.dot/np.matmul, np.linalg.inv, +/np.add, -, * (element-wise), **

# TODO: enumerate operators

# TODO: define derivatives

# C = A + B -> A_a += 1 * C_a; B_a += C_a
# C = A * B -> A_a += B_a * C_a; ...
# C = A * A -> A_a += 2 * A * C_a


class WithRespectTo(Enum):
    Left = 0
    Right = 1


def derivative_UnOp(node: ast.UnaryOp) -> ast.Expr:
    match type(node.op):
        case ast.USub:  # b = -a
            return ast.Expr(value=ast.Constant(value=-1.0))


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
            return ast.Expr(
                value=(node.right if wrt is WithRespectTo.Left else node.left)
            )
        case ast.Div:  # c = a / b -> 1/b or -a / b**2
            return ast.Expr(
                value=(
                    ast.BinOp(
                        op=ast.Div(), left=ast.Constant(value=1.0), right=node.right
                    )
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
        case ast.MatMult:  # c = a @ b
            return ast.Expr(
                value=(node.right if wrt is WithRespectTo.Left else node.left)
            )
        case _:
            raise TypeError("Not implemented yet")


def derivative_Call(call: ast.Call, wrt_arg: int) -> ast.Call:
    func = call.func.attr  # e.g. "exp"

    match func:
        case "exp":
            return call
        case "divide":  # a / b
            if wrt_arg == 0:  # 1 / b
                recip = deepcopy(call.func)
                recip.attr = "reciprocal"  # does not work for integers!

                return ast.Call(func=recip, args=[call.args[1]], keywords=[])

            else:  # - a / b ** 2
                minus_a = ast.UnaryOp(op=ast.USub(), operand=call.args[0])
                b_squared = deepcopy(call)
                b_squared.func.attr = "square"
                b_squared.args = [call.args[1]]
                divide = deepcopy(call.func)
                divide.attr = "divide"

                return ast.Call(func=divide, args=[minus_a, b_squared], keywords=[])
        case "ones":
            return None
        case "zeros":
            return None
        case _:
            raise ValueError("Not implemented yet")
    # TODO: replace deepcopy (slow) with some method to generate ast.Call
