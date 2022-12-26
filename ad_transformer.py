import ast
from copy import deepcopy


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
    # rev_counter = 0  # global counter for reverse section
    var_table = None  # dictionary for transformed variable names i.e. {"A" : "v0"}
    # rev_mode = False  # flag for reverse (adjoint) section

    return_list = None

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
        Generates and inserts primal (forward) single assignment code for given right hand side.
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
        Generates and inserts adjoint (reverse) single assignment code for given right hand side.
        """
        # TODO: flag for incremental?
        new_v = ast.Name(id="v{}_a".format(target_v), ctx=ast.Store())
        new_node = ast.Assign(targets=[new_v], value=rhs)
        self.reverse_stack.insert(
            0, new_node
        )  # always insert at index 0 to ensure reverse order
        return new_v

    def visit_BinOp(self, node: ast.BinOp) -> ast.Name:
        # Not directly visited for A @ B + C as it is contained in Return!
        if isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name):
            # end of recursion:  v_i = A + B
            return self.Generate_BinOp_SAC(node)
        else:
            # visit children recursively to handle nested expressions
            node.left = self.visit(node.left)  # A @ B -> v0
            node.right = self.visit(node.right)  # C
            return self.visit(node)  # recursion

    def Generate_BinOp_SAC(self, binop_node: ast.BinOp) -> ast.Name:
        """Generates and inserts SAC into FunctionDef for BinOp node. Returns the newly assigned variable."""
        new_v = ast.Name(id="v" + str(self.counter), ctx=ast.Store())

        # left and right must have been assigned to some v_i already
        # this could be handled by calling a suitable visit_Name
        left_id = self.var_table.get(binop_node.left.id, binop_node.left.id)
        left_v = ast.Name(id=left_id, ctx=ast.Load())
        right_id = self.var_table.get(binop_node.right.id, binop_node.right.id)
        right_v = ast.Name(id=right_id, ctx=ast.Load())

        new_node = ast.Assign(
            targets=[new_v],
            value=ast.BinOp(left=left_v, op=binop_node.op, right=right_v),
        )
        self.primal_stack.insert(self.counter, new_node)  # insert SAC

        # print("Created {}".format(new_node.targets[0].id))
        self.counter += 1
        return new_v

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.Name:
        """
        Generates a single-assignment forward section
        and an adjoint reverse section of the given function
        """
        self.functionDef = deepcopy(node)  # copy FuncDef to modify it
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

        # print("Final var dict: {}".format(str(self.var_table)))

        # generate reverse section
        # self.rev_counter += self.counter
        # self.counter = 0
        # self.rev_mode = True
        # self.functionDef.args = self.visit(node.args)

        # append finished reverse section
        self.functionDef.body.extend(self.reverse_stack)

        return self.functionDef  # return modified FunctionDef

    def visit_arguments(self, arguments: ast.arguments) -> ast.arguments:
        """Generates SAC for each function argument"""
        # TODO: SAC for arguments in forward mode and appending of adjoint arguments + initialization in reverse mode
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
        final_v = self.visit(node.value)  # i.e. BinOp, Call, ...
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
        # new variable is assigned (some possibly nested expression)
        # replace with v_i node and save old name in dict

        # generate SAC for r.h.s
        new_v = self.visit(node.value)

        if new_v is not None:  # remember v_i
            self.var_table[node.targets[0].id] = new_v.id
        else:  # just insert assignment for now
            new_v = self.SAC(node.value)
        return None

    def visit_Constant(self, node: ast.Constant) -> ast.Name:
        new_v = ast.Name(id="v" + str(self.counter), ctx=ast.Load())
        new_node = ast.Assign(targets=[new_v], value=node)
        self.primal_stack.insert(self.counter, new_node)
        self.counter += 1
        return new_v

    def visit_Call(self, node: ast.Call) -> ast.Name:
        for i in range(len(node.args)):
            arg_v = self.visit(
                node.args[i]
            )  # could simply be a Name or possibly nested BinOp

            node.args[i] = arg_v  # we only need to change the argument

        return self.SAC(node)

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """returns the v_i corresponding to node"""
        return ast.Name(id=self.var_table[node.id], ctx=ast.Load())

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.Name:
        return self.SAC(node)
