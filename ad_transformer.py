import ast
from copy import deepcopy
from derivatives import *


class AdjointNodeTransformer(ast.NodeTransformer):
    """
    This AST transformer takes a given function
    and transforms the primal section to SAC
    and inserts a reverse section with Adjoint code
    """

    functionDef = None  # ast.FunctionDef
    primal_stack = None
    counter = 0  # global SAC counter

    reverse_stack = None
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

    def ad_SAC(self, rhs, target_v: int) -> ast.Name:
        """
        Generates and inserts reverse (adjoint) single assignment code for given right hand side.
        """
        # TODO: flag for incremental?
        new_v = ast.Name(id="v{}_a".format(target_v), ctx=ast.Store())
        new_node = ast.Assign(targets=[new_v], value=rhs)
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
        )
        self.primal_stack.insert(self.counter, new_node)  # insert SAC
        self.counter += 1

        # TODO: adjoints for left and right node
        return new_v

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.Name:
        """
        Generates a single-assignment forward ("primal") section
        and a reverse ("adjoint") section of the given function
        """
        self.functionDef = deepcopy(node)  # copy FuncDef to modify it (efficiency?)
        self.functionDef.name = node.name + "_AD"
        print("Generating {} ...".format(str(self.functionDef.name)))

        # clear the function body
        self.functionDef.body = []

        # recursively visit original function body AST (and update functionDef in the process)
        self.functionDef.args = self.visit(node.args)

        for statements in node.body:
            self.visit(statements)

        # insert SAC forward (primal) section
        self.functionDef.body = self.primal_stack

        # append finished reverse section
        self.functionDef.body.extend(self.reverse_stack)

        return self.functionDef  # return modified FunctionDef

    def visit_arguments(self, arguments: ast.arguments) -> ast.arguments:
        """Generates SAC for each function argument"""
        old_arguments = arguments.args.copy()
        for a in old_arguments:
            # SAC for arguments
            old_v = ast.Name(id=a.arg, ctx=ast.Load())
            self.SAC(old_v)

            # append adjoint arguments
            ad_arg = a.arg + "_a"
            arguments.args.append(ast.arg(arg=ad_arg))
            old_v = ast.Name(id=ad_arg, ctx=ast.Load())
            v_ad = self.ad_SAC(old_v, self.counter - 1)
            self.return_list.append(v_ad)  # remember adjoints for return statement

        return arguments

    def visit_Return(self, node: ast.Return) -> None:
        """generates SAC for expression that is returned and replaces it with
        a return of a tuple of primal results and adjoints of function arguments (derivatives)"""
        final_v = self.visit(node.value)  # i.e. BinOp, Call, ...
        # TODO: initialize adjoint of final v_i

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
            new_v = self.SAC(node.value)

        # TODO: initialize adjoint of new v_i

        return None  # old assignment is removed

    def visit_Constant(self, node: ast.Constant) -> ast.Name:
        """generates SAC for constant assignment and returns the newly assigned v_i"""
        return self.SAC(node)

    def visit_Call(self, node: ast.Call) -> ast.Name:
        """replaces the arguments of a function call, generates SAC, and returns the newly assigned v_i"""
        # TODO: use list comprehension
        for i in range(len(node.args)):
            arg_v = self.visit(
                node.args[i]
            )  # could simply be a Name or possibly nested BinOp

            node.args[i] = arg_v  # we only need to change the argument
        # TODO: derivative of function call

        return self.SAC(node)

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """returns the v_i corresponding to node (lookup)"""
        return ast.Name(id=self.var_table.get(node.id, node.id), ctx=ast.Load())

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.Name:
        """generates SAC for unary operation and returns the newly assigned v_i"""
        return self.SAC(node)
