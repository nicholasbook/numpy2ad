import ast
from .derivatives import *
from .base_transformer import AdjointTransformer


class ExpressionTransformer(AdjointTransformer):
    """
    This AST transformer takes a given expression e.g. "D = A @ B + C",
    transforms the primal section to SAC, and inserts a reverse section with adjoint code.

    Original variable names are left unchanged and their adjoints are assumed to be defined.
    """

    def __init__(self) -> None:
        super().__init__()

    def visit_BinOp(self, node: ast.BinOp) -> ast.Name:
        """Recursively visits binary operation and generates SAC until it resolves to single-variables.

        Args:
            node (ast.BinOp): the BinOp node

        Returns:
            ast.Name: the transformed node
        """

        if isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name):
            # end of recursion:  v_i = A + B

            # preserve name of target for top-level assignments
            new_node = self._make_BinOp_SAC_AD(
                node,
                self.assign_target
                if (self.binop_depth == 0 and self.call_depth == 0)
                else None,
            )
            return new_node
        else:
            return super()._recursive_BinOP(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # variable was incremented by some possibly nested expression
        # old value needs to be recorded! And restored later on
        return super().visit_AugAssign(node)

    def visit_Constant(self, node: ast.Constant) -> ast.Name:
        """generates SAC for constant assignment and returns the newly assigned v_i"""
        new_v = self._generate_SAC(node)
        self._generate_ad_SAC(ast.Constant(value=0.0), self._make_ad_target(new_v), True)
        return new_v

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Visits Module node (root). Visit is forwarded and body is replaced with transformed code.

        Args:
            node (ast.Module): the Module node

        Returns:
            ast.Module: the transformed node
        """
        for expr in node.body:  # visit all nodes
            self.visit(expr)

        # as visit_FunctionDef was not visited so we need to build the output here
        node.body = self.primal_stack
        node.body.extend(self.reverse_init_list)
        node.body.extend(self.reverse_stack)

        return node


def transform_expr(expr: str) -> str:
    """Transforms a given assignment expression to adjoint code.

    Args:
        expr (str): the expression

    Returns:
        str: the transformed expression
    """
    assert isinstance(expr, str)
    transformer = ExpressionTransformer()
    tree = ast.parse(expr)
    assert isinstance(tree.body[0], (ast.Assign, ast.AugAssign))
    transformed_tree = transformer.visit(tree)
    transformed_tree = ast.fix_missing_locations(transformed_tree)
    return ast.unparse(transformed_tree)
