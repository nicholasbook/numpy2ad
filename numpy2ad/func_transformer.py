import ast
from copy import deepcopy
from inspect import getsource
from typing import Callable, Union
from .derivatives import *
from .base_transformer import AdjointTransformer
from copy import copy
from textwrap import dedent

# TODO:
# - factor out as much as possible
# - introduce some basic code optimizations
# - figure out higher derivatives
# - solve for inverse matrix


class FunctionTransformer(AdjointTransformer):
    """
    This AST transformer takes a given function
    and transforms the primal section to SAC
    and inserts a reverse section with Adjoint code.
    """

    functionDef: ast.FunctionDef = None
    return_list: list[ast.Name] = None  # list of values to return
    out_id = "out"

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
            # TODO: or ast.Attribute? e.g. A.T + B
            # TODO: ast.Constant is also okay e.g. 2 * A

            target = None
            if self.binop_depth == 0 and self.call_depth == 0:
                if self.return_target is not None and isinstance(
                    self.return_target, ast.BinOp
                ):
                    # returning a BinOP -> use out
                    assert self.assign_target is None
                    target = self._make_ctx_load(ast.Name(id=self.out_id))
                elif self.assign_target is not None:
                    # assigning a BinOp -> use return_target
                    assert self.return_target is None
                    target = self.assign_target

            return self._make_BinOp_SAC_AD(node, target)
        else:
            return super()._recursive_BinOP(node)

    def visit_Call(self, node: ast.Call) -> ast.Name:
        # special case for function we do not differentiate, e.g. np.eye, np.full, ...
        if derivative_Call(node, 0) is None:
            new_v: ast.Name
            new_v_a: ast.Name
            if (
                self.assign_target is not None
                and self.call_depth == 0
                and self.binop_depth == 0
            ):
                assert self.return_target is None
                new_v = self.assign_target
                self.primal_stack.append(ast.Assign(targets=[new_v], value=node))
                new_v_a = self._generate_ad_SAC(
                    self._numpy_zeros(new_v), self._make_ad_target(new_v), True
                )
                self.assign_target = None
            elif self.return_target is not None and isinstance(self.return_target, ast.Call):
                raise NotImplementedError
                assert self.assign_target is None
                new_v = ast.Name(id=self.out_id, ctx=ast.Store())
                new_v_a = self._make_ad_target(new_v)  # out_a is given as input
                self._generate_custom_SAC(lhs=new_v, rhs=node)
            else:
                new_v = self._generate_SAC(node)
                new_v_a = self._generate_ad_SAC(
                    self._numpy_zeros(new_v), self._make_ad_target(new_v), True
                )
            return new_v  # no point traversing further

        # (recursively) visit arguments
        self.call_depth += 1
        super()._visit_Call_args(node)
        self.call_depth -= 1

        # make (AD) SAC with special cases for assign and return targets
        new_v: ast.Name
        new_v_a: ast.Name
        if (
            self.assign_target is not None and self.call_depth == 0 and self.binop_depth == 0
        ):  # e.g. B = A + np.linalg.inv(C) -> v0 = np.linalg.inv(C); B = A + v0
            assert self.return_target is None
            new_v = self.assign_target
            self.primal_stack.append(ast.Assign(targets=[new_v], value=node))
            new_v_a = self._generate_ad_SAC(
                self._numpy_zeros(new_v), self._make_ad_target(new_v), True
            )
            self.assign_target = None

        elif self.return_target is not None and isinstance(
            self.return_target, ast.Call
        ):  # e.g. return np.linalg.inv(A)
            assert self.assign_target is None
            new_v = ast.Name(id=self.out_id, ctx=ast.Store())
            new_v_a = self._make_ad_target(new_v)  # out_a is given as input
            self._generate_custom_SAC(lhs=new_v, rhs=node)

        else:  # default (nested Call)
            new_v = self._generate_SAC(node)
            new_v_a = self._generate_ad_SAC(
                self._numpy_zeros(new_v), self._make_ad_target(new_v), True
            )

        super()._generate_Call_args_ad_SAC(node, new_v, new_v_a)

        return new_v

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
        if self.return_target is None:
            self.return_target = copy(node.value)
            if isinstance(node.value, ast.Name):
                self.out_id = self.return_target.id

        final_v = self.visit(node.value)  # i.e. BinOp, Call, ...
        assert final_v.id == self.out_id

        # assumes that there is only one output
        self.return_list.insert(0, final_v)

        # append 'out_a' to signature
        out_a_arg = ast.arg(arg=f"{self.out_id}_a")
        if out_a_arg.arg not in [arg.arg for arg in self.functionDef.args.args]:
            self.functionDef.args.args.append(out_a_arg)

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
        # check return value
        if not isinstance(node.body[0], ast.FunctionDef):
            raise ValueError("first line of input must be function definition")

        last_line = node.body[0].body[-1]
        if not isinstance(last_line, ast.Return):
            raise ValueError("return statement must be last line of function")
        if isinstance(last_line.value, ast.Name):
            # self.return_target = last_line.value
            self.out_id = last_line.value.id

        new_node = self.generic_visit(node)

        # insert "import numpy as np"
        new_node.body.insert(0, ast.Import(names=[ast.alias(name="numpy", asname="np")]))
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
        ast.parse(dedent(getsource(func))) if isinstance(func, Callable) else ast.parse(func)
    )
    newAST = transformer.visit(tree)
    newAST = ast.fix_missing_locations(newAST)
    new_code = ast.unparse(newAST)
    if output_file is not None:  # TODO: create __init__.py in target directory?
        file = open(output_file, "w")
        file.write(new_code)
        file.close()
        return
    else:
        return new_code
