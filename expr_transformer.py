import ast
from copy import deepcopy, copy
import re
from inspect import getsource
from typing import Callable, Union

# from derivatives import *

import derivatives
import importlib

importlib.reload(derivatives)
from derivatives import *


class ExpressionAdjointTransformer(ast.NodeTransformer):
    """
    This AST transformer takes a given expression e.g. D = A @ B + C
    and transforms the primal section to SAC
    and inserts a reverse section with Adjoint code
    """

    primal_stack = None  # forward section
    counter = 0  # global SAC counter
    assign_target = None
    binop_depth = 0

    reverse_stack = None  # reverse section
    reverse_init_list = None  # adjoint initialization section
    var_table = None  # dictionary for transformed variable names i.e. {"A" : "v0"}

    def __init__(self) -> None:
        if self.var_table is None:
            self.var_table = dict()
        if self.primal_stack is None:
            self.primal_stack = list()
        if self.reverse_stack is None:
            self.reverse_stack = list()
        if self.reverse_init_list is None:
            self.reverse_init_list = list()

    def _get_id(self, node: ast.Name) -> str:
        res = re.search(r"v(\d+)", node.id)
        if res is None:
            return None
        else:
            return res.group(1)

    def _make_attribute(self, var: str, attr: str) -> ast.Attribute:
        return ast.Attribute(
            value=ast.Name(id=var, ctx=ast.Load()), attr=attr, ctx=ast.Load()
        )

    def _make_call(self, func: ast.Attribute, args: list[ast.Attribute]) -> ast.Call:
        return ast.Call(func=func, args=args, keywords=[])

    def _numpy_zeros(self, var: str) -> str:
        return self._make_call(
            func=self._make_attribute(var="numpy", attr="zeros"),
            args=[self._make_attribute(var=var, attr="shape")],
        )

    def _make_ctx_load(self, node: ast.Name) -> ast.Name:
        new_node = copy(node)
        new_node.ctx = ast.Load()
        return new_node

    def _make_ad_target(self, node: ast.Name) -> ast.Name:
        target = self._get_id(node)
        if target is None:
            return ast.Name(id=node.id + "_a", ctx=ast.Store())  # e.g. A_a
        else:
            return ast.Name(id="v{}_a".format(target), ctx=ast.Store())

    def _generate_SAC(self, rhs) -> ast.Name:
        """
        Generates and inserts forward (primal) single assignment code for given right hand side.
        """
        new_v = ast.Name(id="v{}".format(self.counter), ctx=ast.Store())
        new_node = ast.Assign(targets=[new_v], value=rhs)
        if hasattr(rhs, "id"):  # not the case for ast.Constant
            self.var_table[rhs.id] = new_v.id
        self.primal_stack.insert(self.counter, new_node)

        self.counter += 1
        return new_v

    def _generate_ad_SAC(self, rhs, target: ast.Name, init_mode: bool) -> ast.Name:
        """
        Generates and inserts reverse (adjoint) single assignment code for given right hand side.
        """
        simplify = False
        aug_op = None
        if isinstance(rhs, ast.UnaryOp):  # simplify "+= +-y_a" to "+-= y_a"
            simplify = True
            aug_op = ast.Add() if isinstance(rhs.op, ast.UAdd) else ast.Sub()
            rhs = rhs.operand

        new_node = (
            ast.Assign(targets=[target], value=rhs)
            if init_mode
            else ast.AugAssign(
                op=(aug_op if simplify else ast.Add()), target=target, value=rhs
            )
        )

        if init_mode:
            self.reverse_init_list.append(new_node)
        else:
            self.reverse_stack.insert(
                0, new_node
            )  # always insert at index 0 to ensure reverse order (consider appending for O(1) complexity)

        return target

    def visit_BinOp(self, node: ast.BinOp) -> ast.Name:
        # Not directly visited for A @ B + C as it is contained in Return!
        if isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name):
            # end of recursion:  v_i = A + B
            new_node = self._make_BinOp_SAC(node)
            return new_node
        else:
            self.binop_depth += 1
            # visit children recursively to handle nested expressions
            node.left = self.visit(node.left)  # e.g. A @ B -> v3
            node.right = self.visit(node.right)  # e.g. C -> v2
            self.binop_depth -= 1
            return self.visit(node)  # recursion

    def _make_BinOp_SAC(self, binop_node: ast.BinOp) -> ast.Name:
        """Generates and inserts SAC into FunctionDef for BinOp node. Returns the newly assigned variable."""
        operator = (
            ast.MatMult() if isinstance(binop_node.op, ast.MatMult) else ast.Mult()
        )
        # left and right must have been assigned to some v_i already
        left_v = self.visit(binop_node.left)  # calls visit_Name
        right_v = self.visit(binop_node.right)

        new_v = (
            self.assign_target
            if self.binop_depth == 0
            else ast.Name(id="v" + str(self.counter), ctx=ast.Store())
        )
        # to leave C = A + B unchanged

        new_node = ast.Assign(
            targets=[new_v],
            value=ast.BinOp(left=left_v, op=binop_node.op, right=right_v),
        )  # e.g. v3 = v0 @ v1
        self.primal_stack.insert(self.counter, new_node)  # insert SAC
        self.var_table[new_v.id] = new_v.id

        # initialize adjoint of new v_i
        value = self._make_call(
            func=self._make_attribute(var="numpy", attr="zeros"),
            args=[self._make_attribute(var=new_v.id, attr="shape")],
        )
        new_v_a = (
            self._make_ad_target(new_v)
            if self.binop_depth == 0
            else self._generate_ad_SAC(value, self._make_ad_target(new_v), True)
        )

        # generate derivative of BinOp: left_a/right_a += f' lhs_a
        new_v_a_c = self._make_ctx_load(new_v_a)

        def __make_rhs(deriv: ast.Expr, swap=False):
            if isinstance(deriv.value, ast.Constant):  # simplify +-1.0 * y_a to +-y_a
                return ast.UnaryOp(
                    op=(ast.UAdd() if deriv.value.value > 0 else ast.USub()),
                    operand=new_v_a_c,
                )
            else:
                return ast.BinOp(
                    op=operator,
                    left=deriv.value if not swap else new_v_a_c,
                    right=new_v_a_c if not swap else deriv.value,
                )  # swap order in case of C=AB -> A_a=B^T C_a

        l_deriv = derivative_BinOp(new_node.value, WithRespectTo.Left)  # ast.Expr
        self._generate_ad_SAC(
            rhs=__make_rhs(l_deriv, swap=True),
            target=self._make_ad_target(left_v),
            init_mode=False,
        )

        # rhs of BinOp
        r_deriv = derivative_BinOp(new_node.value, WithRespectTo.Right)  # ast.Expr
        self._generate_ad_SAC(__make_rhs(r_deriv), self._make_ad_target(right_v), False)

        self.counter += 1
        return new_v

    def visit_Assign(self, node: ast.Assign) -> None:
        """generates SAC for an arbitrary assignment in function body"""
        # variable was assigned a possibly nested expression
        # replace with v_i node and save old name in dict

        # for expression mode, this is the entry point.
        self.assign_target = node.targets[0]

        # generate SAC for r.h.s recursively
        new_v = self.visit(node.value)
        # remember v_i
        self.var_table[self.assign_target.id] = new_v.id

        self.assign_target = None

        # TODO: initialize adjoint of new v_i?
        # c = a + b -> assign(binop) -> binop handles intialization (c_a required for ad_SAC anyway)
        # c = a -> aliasing! who does a_a = c_a ?
        # c = f(a) -> assign(call) -> call inits.
        # if rhs is Name (?)
        # self.ad_SAC(ast.Constant(value=0.0), self.get_id(new_v), True)

        return None  # old assignment is removed

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # variable was incremented by some possibly nested expression
        # old value needs to be recorded! And restored later on
        return super().visit_AugAssign(node)

    def visit_Constant(self, node: ast.Constant) -> ast.Name:
        """generates SAC for constant assignment and returns the newly assigned v_i"""
        new_v = self._generate_SAC(node)
        self._generate_ad_SAC(
            ast.Constant(value=0.0), self._make_ad_target(new_v), True
        )
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
                node.args[i] = self._make_ctx_load(arg_v)
            else:
                node.args[i] = arg_v  # replace the argument names

        # SAC for lhs
        new_v = self._generate_SAC(node)

        # initialize adjoint of lhs
        new_v_a = self._generate_ad_SAC(
            self._numpy_zeros(new_v.id), self._make_ad_target(new_v), True
        )

        # adjoints of args (already initialized)
        for i in range(len(node.args)):
            if type(node.args[i]) is ast.Name:
                d = derivative_Call(node, i).value
                if node.func.attr == "inv":  # special case
                    d: ast.UnaryOp
                    d.operand.right.left.id = new_v_a.id
                    self._generate_ad_SAC(
                        rhs=d,
                        target=self._make_ad_target(node.args[i]),
                        init_mode=False,
                    )
                elif d is not None:
                    prod = ast.BinOp(
                        op=ast.Mult(), left=d, right=self._make_ctx_load(new_v_a)
                    )
                    self._generate_ad_SAC(
                        rhs=prod,
                        target=self._make_ad_target(node.args[i]),
                        init_mode=False,
                    )

        return new_v

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """returns the v_i corresponding to node (lookup)"""
        v_id = self.var_table.get(node.id, node.id)
        return ast.Name(id=v_id, ctx=ast.Load())

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.Name:
        """generates SAC for unary operation and returns the newly assigned v_i"""
        op_v = self.visit(node.operand)  # Name, Constant, Call, ...

        # replace operand (variable must be known e.g. -A -> -v0)
        node.operand = self._make_ctx_load(op_v)

        # generate SAC
        new_v = self._generate_SAC(node)

        # generate and initialize adjoint
        # v0 = A, v1 = -A -> v1 = -v0 -> v0_a = A_a, v1_a = 0, v0_a += -v1_a
        new_v_a = self._generate_ad_SAC(
            self._numpy_zeros(new_v.id), self._make_ad_target(new_v), True
        )
        new_v_a_c = copy(new_v_a)
        new_v_a_c.ctx = ast.Load()
        prod = ast.BinOp(
            op=ast.Mult(), left=derivative_UnOp(node).value, right=new_v_a_c
        )
        self._generate_ad_SAC(prod, self._make_ad_target(node.operand), False)

        return new_v

    def visit_Attribute(self, node: ast.Attribute) -> ast.Name:
        if node.attr == "T":
            if isinstance(node.value, ast.Name):  # A^T
                new_v = self._generate_SAC(node)
                self._generate_ad_SAC(
                    rhs=self._make_attribute(var=node.value.id + "_a", attr="T"),
                    target=self._make_ad_target(new_v),
                    init_mode=True,
                )  # e.g. v0_a = A_a^T
                return new_v
            else:  # recursion
                node.value = self.visit(node.value)
                return self.visit(node)
        else:
            return node
        # TODO: support for (A @ B).T (Attribute(BinOp))

    def visit_Module(self, node: ast.Module) -> ast.Module:
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
        expr (str): _description_

    Returns:
        list: _description_
    """
    assert isinstance(expr, str)
    transformer = ExpressionAdjointTransformer()
    tree = ast.parse(expr)
    assert isinstance(tree.body[0], ast.Assign)  # or AugAssign
    transformed_tree = transformer.visit(tree)
    transformed_tree = ast.fix_missing_locations(transformed_tree)
    return ast.unparse(transformed_tree)


# TODO:
# Implement transformer that works on a single expression i.e. D = A@B+C.
# The goal is to make it work recursively on the output expressions.
# Add an option for output to file. Make numpy include optional.
# Write some unit tests.
