import ast
from copy import copy
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

    binop_depth: int = 0  # nested BinOps counter
    call_depth = 0  # nested Calls counter
    assign_target = None  # target of ast.Assign
    return_target = None  # target of ast.Return

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

    def _make_attribute(self, value: str | ast.Attribute, attr: str) -> ast.Attribute:
        if isinstance(value, ast.Attribute):
            return ast.Attribute(value=value, attr=attr, ctx=ast.Load())
        else:
            return ast.Attribute(
                value=ast.Name(id=value, ctx=ast.Load()), attr=attr, ctx=ast.Load()
            )

    def _make_call(self, func: ast.Attribute, args: list[ast.Attribute]) -> ast.Call:
        return ast.Call(func=func, args=args, keywords=[])

    def _numpy_zeros(self, var: ast.Name) -> str:
        return self._make_call(
            func=self._make_attribute(value="np", attr="zeros_like"),
            args=[self._make_ctx_load(var)],
        )  # this does not work for scalars !! # TODO: use np.zeros_like

    def _make_ctx_load(self, node: ast.Name) -> ast.Name:
        new_node = copy(node)
        new_node.ctx = ast.Load()
        return new_node

    def _make_ctx_store(self, node: ast.Name) -> ast.Name:
        new_node = copy(node)
        new_node.ctx = ast.Store()
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
        self.primal_stack.append(new_node)

        self.counter += 1
        return new_v

    def _generate_custom_SAC(self, lhs: ast.Name, rhs: ast) -> None:
        new_sac = ast.Assign(
            targets=[self._make_ctx_store(lhs)], value=self._make_ctx_store(rhs)
        )
        self.primal_stack.append(new_sac)

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

    def _recursive_BinOP(self, node: ast.BinOp):
        # visit children recursively to handle nested expressions
        self.binop_depth += 1
        node.left = self.visit(node.left)  # e.g. A @ B -> v3
        node.right = self.visit(node.right)  # e.g. C -> v2
        self.binop_depth -= 1
        return self.visit(node)  # recursion

    def _make_BinOp_SAC_AD(self, binop_node: ast.BinOp, target: ast.Name = None) -> ast.Name:
        """Generates and inserts SAC into FunctionDef for BinOp node. Returns the newly assigned variable.

        Args:
            binop_node (ast.BinOp): the binary operation\n
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
        self.primal_stack.append(new_node)  # insert SAC
        self.var_table[new_v.id] = new_v.id

        # adjoint
        new_v_a = self._make_ad_target(new_v)
        if target is not None:
            if hasattr(self, "out_id"):
                if (
                    not target.id == self.out_id
                ):  # initialize special target only if it is not the output
                    self._generate_ad_SAC(self._numpy_zeros(new_v), new_v_a, True)
            else:
                pass  # do not initialize (expression mode)
        else:  # default
            self._generate_ad_SAC(self._numpy_zeros(new_v), new_v_a, True)

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
        if r_deriv is not None:
            self._generate_ad_SAC(__make_rhs(r_deriv), self._make_ad_target(right_v), False)

        if target is None:
            self.counter += 1
        return new_v

    def visit_Constant(self, node: ast.Constant) -> ast.Name:
        """generates SAC for constant assignment and returns the newly assigned v_i"""
        # TODO: keep BinOps like '2 * A' as is
        # in which cases do we even need SAC for constants?
        if isinstance(node.value, str):
            return node  # docstring is removed
        new_v = self._generate_SAC(node)
        self._generate_ad_SAC(ast.Constant(value=0.0), self._make_ad_target(new_v), True)
        return new_v

    def _visit_Call_args(self, node: ast.Call) -> None:
        # TODO: use list comprehension ?
        for i in range(len(node.args)):
            arg_v = self.visit(
                node.args[i]
            )  # could simply be a Name or possibly nested BinOp, Call, ...

            node.args[i] = self._make_ctx_load(arg_v)  # replace the argument names

    def _generate_Call_args_ad_SAC(
        self, node: ast.Call, new_v: ast.Name, new_v_a: ast.Name
    ) -> None:
        # adjoints of args (already initialized)
        for i in range(len(node.args)):
            if isinstance(node.args[i], ast.Name):
                d = derivative_Call(node, i)
                if d is not None:
                    deriv = d.value

                    if isinstance(node.func.value, ast.Attribute):
                        if node.func.value.attr == "linalg" and node.func.attr == "inv":
                            # e.g. v = np.linalg.inv(A) -> A_a -= v.T @ v_a @ v
                            assert len(node.args) == 1

                            deriv: ast.UnaryOp
                            deriv.operand.left.value.id = new_v.id  # insert v
                            deriv.operand.right.left.id = new_v_a.id  # insert v_a
                            deriv.operand.right.right.id = new_v.id

                            self._generate_ad_SAC(
                                rhs=deriv,
                                target=self._make_ad_target(node.args[i]),
                                init_mode=False,
                            )
                    elif deriv is not None:
                        prod = ast.BinOp(
                            op=ast.Mult(), left=deriv, right=self._make_ctx_load(new_v_a)
                        )
                        self._generate_ad_SAC(
                            rhs=prod,
                            target=self._make_ad_target(node.args[i]),
                            init_mode=False,
                        )

    def visit_Call(self, node: ast.Call) -> ast.Name:
        """replaces the arguments of a function call, generates SAC, and returns the newly assigned v_i"""
        self.call_depth += 1
        self._visit_Call_args(node)
        self.call_depth -= 1

        # make (AD) SAC with special cases for assign targets
        new_v: ast.Name
        new_v_a: ast.Name
        if (
            self.assign_target is not None and self.call_depth == 0 and self.binop_depth == 0
        ):  # e.g. B = A + np.linalg.inv(C) -> v0 = np.linalg.inv(C); B = A + v0
            new_v = self.assign_target
            self.primal_stack.append(ast.Assign(targets=[new_v], value=node))
            new_v_a = self._generate_ad_SAC(
                self._numpy_zeros(new_v), self._make_ad_target(new_v), True
            )
            self.assign_target = None

        else:  # default (nested Call)
            new_v = self._generate_SAC(node)
            new_v_a = self._generate_ad_SAC(
                self._numpy_zeros(new_v), self._make_ad_target(new_v), True
            )

        self._generate_Call_args_ad_SAC(node, new_v, new_v_a)

        return new_v

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """returns the v_i corresponding to node (lookup)"""
        v_id = self.var_table.get(node.id, node.id)  # defaults to id if not found in dict
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
        prod = ast.BinOp(op=ast.Mult(), left=derivative_UnOp(node).value, right=new_v_a_c)
        self._generate_ad_SAC(prod, self._make_ad_target(node.operand), False)

        return new_v

    def visit_Attribute(self, node: ast.Attribute) -> ast.Name:
        if node.attr == "T":
            if isinstance(node.value, ast.Name):  # A^T
                orig_id = node.value.id
                new_v = self.var_table.get(orig_id + "_T", None)  # lookup
                if new_v is not None:
                    return new_v
                else:
                    new_v = self._generate_SAC(node)
                    rhs = (  # self._make_call(func=self._make_attribute(value=
                        self._make_attribute(value=orig_id + "_a", attr="T")
                    )  # , attr="copy"), args=[])
                    self._generate_ad_SAC(
                        rhs=rhs,
                        target=self._make_ad_target(new_v),
                        init_mode=True,
                    )  # e.g. v0_a = A_a.T
                    self.var_table[orig_id + "_T"] = new_v  # store in table
                    return new_v
            else:  # recursion
                node.value = self.visit(node.value)
                return self.visit(node)
        elif node.attr == "shape":
            return None  # ignore
        else:
            return node

    def visit_Assign(self, node: ast.Assign) -> None:
        """generates SAC for an arbitrary assignment in function body"""
        # variable was assigned a possibly nested expression
        # replace with v_i node and save old name in dict
        assert len(node.targets) == 1

        # we want to preserve simple assignments e.g. C = A + B
        # also in the case of nested rhs e.g. D = A @ B + C -> v0 = A @ B; D = v0 + C
        # this requires custom logic to keep certain variable names unchanged
        # in general BinOps / Calls generate their own (AD) SAC (context-free)
        # so we store 'D' for the top level rhs expression as 'assign_target'

        self.assign_target = copy(node.targets[0])

        # TODO: check for simple assignment e.g. B = A
        # generate SAC for r.h.s (e.g. BinOp, Call, ...) recursively
        new_v = self.visit(node.value)

        self.assign_target = None  # reset

        return None  # the original assignment is removed

    def visit_AugAssign(self, node: ast.AugAssign):
        # overwriting of variables must be dealt with
        # e.g. A += B <=> A = A + B => v = A; A = v + B => B_a += A_a; v_a += A_a; A_a += v_a
        raise NotImplementedError()
        # if isinstance(node.op, ast.Add):  # A += A
        #     rhs = ast.BinOp(op=ast.Add(), left=self._make_ctx_load(
        #         node.target), right=node.value)  # A + A
        #     new_v = self.visit(rhs)  # generate SAC
        #     self.var_table[node.target.id] = new_v.id  # ??
        # return None  # remove?
