import ast
from copy import deepcopy
from inspect import getsource
from typing import Callable, Union
from .derivatives import *
from .base_transformer import AdjointTransformer

# TODO:
# - factor out as much as possible
# - introduce some basic code optimizations
# - figure out higher derivatives
# - solve for inverse matrix
# - consider using input variable names (maybe by a secondary pass over AST)


class FunctionTransformer(AdjointTransformer):
    """
    This AST transformer takes a given function
    and transforms the primal section to SAC
    and inserts a reverse section with Adjoint code.
    """

    functionDef: ast.FunctionDef = None
    return_list: list[ast.Name] = None  # list of values to return

    def __init__(self) -> None:
        super().__init__()
        self.return_list = list()

    def visit_BinOp(self, node: ast.BinOp) -> ast.Name:
        """Recursively visits binary operation and generates SAC until it resolves to single-variables.

        Args:
            node (ast.BinOp): the BinOp node

        Returns:
            ast.Name: the transformed node
        """
        if isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name):
            # end of recursion:  v_i = A + B
            return self._make_BinOp_SAC_AD(
                node,
                self._make_ctx_load(ast.Name(id="out"))
                if self.binop_depth == 0 and isinstance(self.return_target, ast.BinOp)
                else None,
            )
        else:
            # visit children recursively to handle nested expressions
            self.binop_depth += 1
            node.left = self.visit(node.left)  # e.g. A @ B -> v3
            node.right = self.visit(node.right)  # e.g. C -> v2
            self.binop_depth -= 1
            return self.visit(node)  # recursion

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.Name:
        """
        Generates a single-assignment forward ("primal") section
        and a reverse ("adjoint") section of the given function
        """

        self.functionDef = ast.FunctionDef(
            name=node.name + "_ad",
            args=deepcopy(node.args),
            body=[],
            decorator_list=[],
            returns=[],
        )

        # clear the function body
        self.functionDef.body = []

        # recursively visit original function body AST (and update functionDef in the process)
        self.functionDef.args = self.visit(node.args)

        for statements in node.body:
            self.visit(statements)

        # insert SAC forward (primal) section
        self.functionDef.body = self.primal_stack

        # append finished reverse section
        self.functionDef.body.extend(self.reverse_init_list)
        self.functionDef.body.extend(self.reverse_stack)

        return self.functionDef  # return modified FunctionDef

    def visit_arguments(self, arguments: ast.arguments) -> ast.arguments:
        """Generates SAC for each function argument"""
        old_arguments = arguments.args.copy()
        for a in old_arguments:
            # store in symbol table
            self.var_table[a.arg] = a.arg

            # append adjoint arguments
            ad_arg = a.arg + "_a"
            arguments.args.append(ast.arg(arg=ad_arg))

            # remember adjoints for return statement
            self.return_list.append(self._make_ctx_load(ast.Name(id=ad_arg)))

            # TODO: add type hints

        return arguments

    def visit_Return(self, node: ast.Return) -> None:
        """generates SAC for expression that is returned and replaces it with
        a return of a tuple of primal results and adjoints of function arguments (derivatives)
        """
        # self.final_assign = True
        self.return_target = node.value
        final_v = self.visit(node.value)  # i.e. BinOp, Call, ...
        final_v.id = "out"

        # assumes that there is only one output
        self.return_list.insert(0, final_v)

        self.functionDef.args.args.append(ast.arg(arg="out_a"))

        return_list = [ast.Name(id=arg.id, ctx=ast.Load()) for arg in self.return_list]

        new_return = ast.Tuple(
            elts=return_list,
            ctx=ast.Load(),
        )
        final_return = ast.Return(value=new_return)
        self.reverse_stack.append(final_return)
        # TODO: type hints

        return None  # original Return is not used

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Visits Module node (root). Visit is forwarded and numpy import statement is inserted.

        Returns:
            ast.Module: the transformed node
        """
        new_node = self.generic_visit(node)
        # insert "import numpy"
        new_node.body.insert(
            0, ast.Import(names=[ast.alias(name="numpy")])
        )  # 'as np' ?
        return new_node


def transform(func: Union[Callable, str], output_file: str = None) -> str:
    """
    Transforms the source code of a given function to include
    the computation of derivatives using reverse ("adjoint") mode automatic differentiation.
    The transformed function is returned as a string that can be compiled and executed.

    The forward ("primal") section is tranformed to single-assignment code
    and the adjoints of all input variables are returned togethere with the primal result.
    """
    transformer = FunctionTransformer()
    tree = (
        ast.parse(getsource(func)) if isinstance(func, Callable) else ast.parse(func)
    )  # does not seem to work for indended code
    newAST = transformer.visit(tree)
    newAST = ast.fix_missing_locations(newAST)
    new_code = ast.unparse(newAST)
    if output_file is not None:  # TODO: create __init__.py in target directory
        file = open(output_file, "w")
        file.write(new_code)
        file.close()
        return
    else:
        return new_code
