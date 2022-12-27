import numpy as np
import ast
from enum import Enum

# TODO: @/np.dot/np.matmul, np.linalg.inv, +/np.add, -, * (element-wise), **

# TODO: enumerate operators

# TODO: define derivatives

# C = A + B -> A_a += 1 * C_a; B_a += C_a
# C = A * B -> A_a += B_a * C_a; ...
# C = A * A -> A_a += 2 * A * C_a


class WithRespectTo(Enum):
    Left = 0
    Right = 1


def derivative_BinOp(node: ast.BinOp, wrt: WithRespectTo) -> ast.Expr:
    """
    Returns an expression that contains the partial derivative
    of the given binary operation with respect to the specified argument (Left or Right).
    """
    match type(node.op):
        case ast.Add:  # c = l + r
            return ast.Expr(value=ast.Constant(value=1))
        case ast.Sub:  # c = l - r
            return ast.Expr(
                value=ast.Constant(value=(1 if wrt is WithRespectTo.Left else -1))
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
                value=(node.left if wrt is WithRespectTo.Left else node.right)
            )
        case _:
            raise TypeError("Not implemented yet")
