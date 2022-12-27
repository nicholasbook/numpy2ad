import numpy as np
import ast

# TODO: @/np.dot/np.matmul, np.linalg.inv, +/np.add, -, * (element-wise), **

# TODO: enumerate operators

# TODO: define derivatives

# C = A + B -> A_a += 1 * C_a; B_a += C_a
# C = A * B -> A_a += B_a * C_a; ...
# C = A * A -> A_a += 2 * A * C_a


def derivative(A, B, operation):
    return
    # match type(operation):
    #     case ast.Add:
    #         return ast.Add()
    #     case ast.Mult:
    #         return
