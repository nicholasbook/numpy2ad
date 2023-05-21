import ast
from copy import copy, deepcopy
import re
from .derivatives import *


class AdjointTransformer(ast.NodeTransformer):
    """Base class for other adjoint code transformers."""

    # TODO: docstring
    var_table: dict = None  # dictionary for transformed variables i.e. "A" : "v0"
    primal_stack: list = None  # SAC code
    counter: int = 0  # SAC counter for v_i
    reverse_stack: list = None  # adjoint code
    reverse_init_list: list = None  # adjoint initialization code

    binop_depth: int = 0  # recursion counter
    return_target = None

    def __init__(self) -> None:
        self.var_table = dict()
        self.primal_stack = list()
        self.reverse_stack = list()
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

    def _numpy_zeros(self, var: ast.Name) -> str:
        return self._make_call(
            func=self._make_attribute(var="numpy", attr="zeros_like"),
            args=[self._make_ctx_load(var)],
        )  # this does not work for scalars !! # TODO: use np.zeros_like

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

        # TODO: consider removing init list and instead using a non-incremental assignment the first time an adjoint variable is used
        if init_mode:
            self.reverse_init_list.append(new_node)
        else:
            self.reverse_stack.insert(
                0, new_node
            )  # always insert at index 0 to ensure reverse order (consider appending for O(1) complexity)

        return target

    def _make_BinOp_SAC_AD(
        self, binop_node: ast.BinOp, target: ast.Name = None
    ) -> ast.Name:
        """Generates and inserts SAC into FunctionDef for BinOp node. Returns the newly assigned variable.

        Args:
            binop_node (ast.BinOp): the binary operation
            target (ast.Name): target node for SAC. If not specified, v{counter} is used.
        """
        operator, swap = (
            (ast.MatMult(), True)
            if isinstance(binop_node.op, ast.MatMult)
            else (ast.Mult(), False)
        )
        # left and right must have been assigned to some v_i already
        left_v = self.visit(binop_node.left)  # calls visit_Name
        right_v = self.visit(binop_node.right)

        new_v = (
            target
            if target is not None
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
        value = self._numpy_zeros(new_v)

        new_v_a = (
            self._make_ad_target(new_v)
            if target is not None
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
            rhs=__make_rhs(l_deriv, swap=swap),
            target=self._make_ad_target(left_v),
            init_mode=False,
        )

        # rhs of BinOp
        r_deriv = derivative_BinOp(new_node.value, WithRespectTo.Right)  # ast.Expr
        self._generate_ad_SAC(__make_rhs(r_deriv), self._make_ad_target(right_v), False)

        self.counter += 1
        return new_v

    def visit_Constant(self, node: ast.Constant) -> ast.Name:
        """generates SAC for constant assignment and returns the newly assigned v_i"""
        if isinstance(node.value, str):
            return node  # docstring is removed
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
            if isinstance(arg_v.ctx, ast.Store):
                node.args[i] = self._make_ctx_load(arg_v)
            else:
                node.args[i] = arg_v  # replace the argument names

        # SAC for lhs
        new_v: ast.Name = self._generate_SAC(node)
        init = True
        if self.return_target is not None:
            new_v.id = "out"
            init = False

        # initialize adjoint of lhs
        new_v_a = self._generate_ad_SAC(
            self._numpy_zeros(new_v), self._make_ad_target(new_v), init
        )

        # adjoints of args (already initialized)
        for i in range(len(node.args)):
            if isinstance(node.args[i], ast.Name):
                d = derivative_Call(node, i).value

                if node.func.value.attr == "linalg" and node.func.attr == "inv":
                    # e.g. v = np.linalg.inv(A) -> A_a -= v.T @ v_a @ v
                    assert len(node.args) == 1

                    d: ast.UnaryOp
                    d.operand.left.value.id = new_v.id  # insert v
                    d.operand.right.left.id = new_v_a.id  # insert v_a
                    d.operand.right.right.id = new_v.id

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
        v_id = self.var_table.get(
            node.id, node.id
        )  # defaults to id if not found in dict
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
            self._numpy_zeros(new_v), self._make_ad_target(new_v), True
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
                )  # e.g. v0_a = A_a.T
                return new_v
            else:  # recursion
                node.value = self.visit(node.value)
                return self.visit(node)
        else:
            return node

    def visit_Assign(self, node: ast.Assign) -> None:
        """generates SAC for an arbitrary assignment in function body"""
        # variable was assigned a possibly nested expression
        # replace with v_i node and save old name in dict
        assert len(node.targets) == 1

        # generate SAC for r.h.s recursively
        new_v = self.visit(node.value)
        # remember v_i
        self.var_table[node.targets[0].id] = new_v.id

        return None  # old assignment is removed

    def visit_AugAssign(self, node: ast.AugAssign):
        # overwriting of variables must be dealt with
        # e.g. A += B <=> A = A + B => v = A; A = v + B => B_a += A_a; v_a += A_a; A_a += v_a
        pass
        # if isinstance(node.op, ast.Add):  # A += A
        #     rhs = ast.BinOp(op=ast.Add(), left=self._make_ctx_load(
        #         node.target), right=node.value)  # A + A
        #     new_v = self.visit(rhs)  # generate SAC
        #     self.var_table[node.target.id] = new_v.id  # ??
        # return None  # remove?
