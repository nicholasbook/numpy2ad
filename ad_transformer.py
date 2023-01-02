import ast
from copy import deepcopy, copy
import re
from inspect import getsource
from typing import Callable

# from derivatives import *

import derivatives
import importlib

importlib.reload(derivatives)
from derivatives import *


class AdjointNodeTransformer(ast.NodeTransformer):
    """
    This AST transformer takes a given function
    and transforms the primal section to SAC
    and inserts a reverse section with Adjoint code
    """

    functionDef = None  # ast.FunctionDef
    primal_stack = None  # forward section
    counter = 0  # global SAC counter

    reverse_stack = None  # reverse section
    reverse_init_list = None  # adjoint initialization section
    var_table = None  # dictionary for transformed variable names i.e. {"A" : "v0"}
    return_list = None  # list of values to return

    def __init__(self) -> None:
        if self.var_table is None:
            self.var_table = dict()
        if self.primal_stack is None:
            self.primal_stack = list()
        if self.reverse_stack is None:
            self.reverse_stack = list()
        if self.return_list is None:
            self.return_list = list()
        if self.reverse_init_list is None:
            self.reverse_init_list = list()

    def get_id(self, node: ast.Name) -> str:
        return re.search(r"v(\d+)", node.id).group(1)

    def generate_attribute(self, var: str, attr: str) -> ast.Attribute:
        return ast.Attribute(
            value=ast.Name(id=var, ctx=ast.Load()), attr=attr, ctx=ast.Load()
        )

    def generate_call(self, func: ast.Attribute, args: list[ast.Attribute]) -> ast.Call:
        return ast.Call(func=func, args=args, keywords=[])

    def numpy_zeros(self, var: str) -> str:
        return self.generate_call(
            func=self.generate_attribute(var="numpy", attr="zeros"),
            args=[self.generate_attribute(var=var, attr="shape")],
        )

    def SAC(self, rhs) -> ast.Name:
        """
        Generates and inserts forward (primal) single assignment code for given right hand side.
        """
        new_v = ast.Name(id="v{}".format(self.counter), ctx=ast.Store())
        new_node = ast.Assign(targets=[new_v], value=rhs)
        if hasattr(rhs, "id"):
            self.var_table[rhs.id] = new_v.id
        self.primal_stack.insert(self.counter, new_node)

        self.counter += 1
        return new_v

    def ad_SAC(self, rhs, target_v: int, init_mode: bool) -> ast.Name:
        """
        Generates and inserts reverse (adjoint) single assignment code for given right hand side.
        """
        new_v = ast.Name(id="v{}_a".format(target_v), ctx=ast.Store())
        new_node = (
            ast.Assign(targets=[new_v], value=rhs)
            if init_mode
            else ast.AugAssign(op=ast.Add(), target=new_v, value=rhs)
        )

        if init_mode:
            self.reverse_init_list.append(new_node)
        else:
            self.reverse_stack.insert(
                0, new_node
            )  # always insert at index 0 to ensure reverse order (consider appending for O(1) complexity)

        return new_v

    def visit_BinOp(self, node: ast.BinOp) -> ast.Name:
        # Not directly visited for A @ B + C as it is contained in Return!
        if isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name):
            # end of recursion:  v_i = A + B
            return self.BinOp_SAC(node)
        else:
            # visit children recursively to handle nested expressions
            node.left = self.visit(node.left)  # e.g. A @ B -> v3
            node.right = self.visit(node.right)  # e.g. C -> v2
            return self.visit(node)  # recursion

    def BinOp_SAC(self, binop_node: ast.BinOp) -> ast.Name:
        """Generates and inserts SAC into FunctionDef for BinOp node. Returns the newly assigned variable."""
        new_v = ast.Name(id="v" + str(self.counter), ctx=ast.Store())

        # left and right must have been assigned to some v_i already
        left_v = self.visit(binop_node.left)  # calls visit_Name
        right_v = self.visit(binop_node.right)

        new_node = ast.Assign(
            targets=[new_v],
            value=ast.BinOp(left=left_v, op=binop_node.op, right=right_v),
        )  # e.g. v3 = v0 @ v1
        self.primal_stack.insert(self.counter, new_node)  # insert SAC

        # initialize adjoint of new v_i
        value = self.generate_call(
            func=self.generate_attribute(var="numpy", attr="zeros"),
            args=[self.generate_attribute(var=new_v.id, attr="shape")],
        )
        new_v_a = self.ad_SAC(value, self.counter, True)

        # generate derivative of BinOp: left_a/right_a += f' lhs_a
        new_v_a_c = copy(new_v_a)
        new_v_a_c.ctx = ast.Load()
        deriv = derivative_BinOp(new_node.value, WithRespectTo.Left)  # ast.Expr
        prod = ast.BinOp(op=ast.Mult(), left=deriv.value, right=new_v_a_c)
        self.ad_SAC(prod, self.get_id(left_v), False)
        deriv = derivative_BinOp(new_node.value, WithRespectTo.Right)  # ast.Expr
        prod = ast.BinOp(op=ast.Mult(), left=deriv.value, right=new_v_a_c)
        self.ad_SAC(prod, self.get_id(right_v), False)

        self.counter += 1
        return new_v

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.Name:
        """
        Generates a single-assignment forward ("primal") section
        and a reverse ("adjoint") section of the given function
        """
        self.functionDef = deepcopy(node)  # copy FuncDef to modify it (efficiency?)
        self.functionDef.name = node.name + "_ad"
        # print("Generating {} ...".format(str(self.functionDef.name)))

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
            # SAC for arguments
            old_var = ast.Name(id=a.arg, ctx=ast.Load())
            self.SAC(old_var)

            # append adjoint arguments
            ad_arg = a.arg + "_a"
            arguments.args.append(ast.arg(arg=ad_arg))

            new_v = ast.Name(id="v{}_a".format(self.counter - 1), ctx=ast.Store())
            old_var_ad = ast.Name(id=ad_arg, ctx=ast.Load())
            self.ad_SAC(old_var_ad, self.counter - 1, True)

            self.return_list.append(new_v)  # remember adjoints for return statement

            # TODO: add type hints

        return arguments

    def visit_Return(self, node: ast.Return) -> None:
        """generates SAC for expression that is returned and replaces it with
        a return of a tuple of primal results and adjoints of function arguments (derivatives)"""
        final_v = self.visit(node.value)  # i.e. BinOp, Call, ...

        self.reverse_init_list[-1].value = self.generate_call(
            func=self.generate_attribute(var="numpy", attr="ones"),
            args=[self.generate_attribute(var=final_v.id, attr="shape")],
        )
        self.return_list.insert(0, final_v)

        return_list = [ast.Name(id=arg.id, ctx=ast.Load()) for arg in self.return_list]

        new_return = ast.Tuple(
            elts=return_list,
            ctx=ast.Load(),
        )
        final_return = ast.Return(value=new_return)
        self.reverse_stack.append(final_return)

        return None  # original Return is not used

    def visit_Assign(self, node: ast.Assign) -> None:
        """generates SAC for an arbitrary assignment in function body"""
        # variable was assigned a possibly nested expression
        # replace with v_i node and save old name in dict

        # generate SAC for r.h.s recursively
        new_v = self.visit(node.value)

        if new_v is not None:  # remember v_i
            self.var_table[node.targets[0].id] = new_v.id
        else:  # just insert assignment for now
            raise ValueError("visit(rhs of Assign) returned None")
            # new_v = self.SAC(node.value)

        # TODO: initialize adjoint of new v_i?
        # c = a + b -> assign(binop) -> binop handles intialization (c_a required for ad_SAC anyway)
        # c = a -> aliasing! who does a_a = c_a ?
        # c = f(a) -> assign(call) -> call inits.
        # if rhs is Name (?)
        # self.ad_SAC(ast.Constant(value=0.0), self.get_id(new_v), True)

        return None  # old assignment is removed

    def visit_Constant(self, node: ast.Constant) -> ast.Name:
        """generates SAC for constant assignment and returns the newly assigned v_i"""
        new_v = self.SAC(node)
        self.ad_SAC(ast.Constant(value=0.0), self.get_id(new_v), True)
        return new_v

    def visit_Call(self, node: ast.Call) -> ast.Name:
        """replaces the arguments of a function call, generates SAC, and returns the newly assigned v_i"""
        # TODO: use list comprehension
        for i in range(len(node.args)):
            arg_v = self.visit(
                node.args[i]
            )  # could simply be a Name or possibly nested BinOp

            # ensure arg_v is loaded
            if type(arg_v.ctx) is ast.Store:
                arg_v_c = copy(arg_v)
                arg_v_c.ctx = ast.Load()
                node.args[i] = arg_v_c
            else:
                node.args[i] = arg_v  # replace the argument names

        # SAC for lhs
        new_v = self.SAC(node)

        # initialize adjoint of lhs
        new_v_a = self.ad_SAC(self.numpy_zeros(new_v.id), self.get_id(new_v), True)

        # adjoints of args (already initialized)
        for i in range(len(node.args)):
            if type(node.args[i]) is ast.Name:
                d = derivative_Call(node, i)
                if d is not None:
                    new_v_a_c = copy(new_v_a)
                    new_v_a_c.ctx = ast.Load()
                    prod = ast.BinOp(op=ast.Mult(), left=d, right=new_v_a_c)
                    self.ad_SAC(prod, self.get_id(node.args[i]), False)

        return new_v

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """returns the v_i corresponding to node (lookup)"""
        return ast.Name(id=self.var_table.get(node.id, node.id), ctx=ast.Load())

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.Name:
        """generates SAC for unary operation and returns the newly assigned v_i"""
        op_v = self.visit(node.operand)  # Name, Constant, Call, ...

        # replace operand (variable must be known e.g. -A -> -v0)
        op_v_c = copy(op_v)
        op_v_c.ctx = ast.Load()
        node.operand = op_v_c

        # generate SAC
        new_v = self.SAC(node)

        # generate and initialize adjoint
        # v0 = A, v1 = -A -> v1 = -v0 -> v0_a = A_a, v1_a = 0, v0_a += -v1_a
        new_v_a = self.ad_SAC(self.numpy_zeros(new_v.id), self.get_id(new_v), True)
        new_v_a_c = copy(new_v_a)
        new_v_a_c.ctx = ast.Load()
        prod = ast.BinOp(
            op=ast.Mult(), left=derivative_UnOp(node).value, right=new_v_a_c
        )
        self.ad_SAC(prod, self.get_id(node.operand), False)

        return new_v

    def visit_Module(self, node: ast.Module) -> ast.Module:
        new_node = self.generic_visit(node)
        # insert "import numpy"
        new_node.body.insert(0, ast.Import(names=[ast.alias(name="numpy")]))
        return new_node


def transform(func: Callable) -> str:
    """
    Transforms the source code of a given function to include
    the computation of derivatives using reverse ("adjoint") mode automatic differentiation.
    The transformed function is given as a string that can be compiled and executed.

    The forward ("primal") section is tranformed to single-assignment code
    and the adjoints of all input variables are returned togethere with the primal result.
    """
    transformer = AdjointNodeTransformer()
    tree = ast.parse(getsource(func))
    newAST = transformer.visit(tree)
    newAST = ast.fix_missing_locations(newAST)
    return ast.unparse(newAST)
    # TODO: make transform(transform(...)) work
