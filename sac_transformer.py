import ast
from copy import deepcopy


def nameof(node):
    """return name of type of node as str"""
    return type(node).__name__


class SACGraphTransformer(ast.NodeTransformer):
    """A recursive node transformer that generates SAC code (for all intermediate results).

    Algorithm:
    Traverse the tree recursively and look for BinOps that are not assigned to a variable.
    New lines of code should be children of FunctionDef.
    Requires a global SAC counter.

    Which nodes need to be transformed? - FunctionDef, BinOp (if not assigned), Call, AugAssign
    Change function name to f_SAC.
    Create annotations for initialization, augmented forward mode, and reverse mode (not possible only with ast)
    New nodes need lineno and col_offset attributes! use ast.fix_missing_locations()

    TODO:
        - clean up, refactor, and document
        - add more AST types?
    """

    functionDef = None
    counter = 0
    var_table = None  # dictionary for transformed variable names i.e. {"A" : "v0"}

    def __init__(self) -> None:
        if self.var_table is None:
            self.var_table = dict()

    def Generate_SAC_recursively(self, node):
        # called for arbitrary BinOp, Return, Assignment, ...
        # TODO: document and visualize why this works

        if isinstance(node, ast.Return):
            self.visit(node.value)  # i.e. BinOp, Call, ...

            self.Generate_Return_SAC(node)  # generate code for final return
        elif isinstance(node, ast.Assign):
            # new variable is assigned (some possibly nested expression)
            # replace with v_i node and save old name in dict

            # generate SAC for r.h.s
            new_v = self.Generate_SAC_recursively(node.value)

            if new_v is not None:  # remember v_i
                self.var_table[node.targets[0].id] = new_v.id
            else:  # just insert assignment for now
                new_v = ast.Name(id="v" + str(self.counter), ctx=ast.Store())
                new_node = ast.Assign(targets=[new_v], value=node.value)
                self.var_table[node.targets[0].id] = new_v.id
                self.functionDef.body.insert(self.counter, new_node)  # insert SAC
                self.counter += 1

            return node
        elif isinstance(node, ast.BinOp):
            # visit children recursively to handle nested expressions
            node.left = self.visit(node.left)  # A @ B -> v0
            node.right = self.visit(node.right)  # C

            return self.Generate_BinOp_SAC(node)  # generate code for root node
        elif isinstance(node, ast.Call):
            # generate SAC for Call arguments
            for i in range(len(node.args)):
                arg_v = self.Generate_SAC_recursively(
                    node.args[i]
                )  # could simply be a Name or possibly nested BinOp

                node.args[i] = arg_v  # we only need to change the argument

            # generate SAC for Call
            new_v = ast.Name(id="v" + str(self.counter), ctx=ast.Store())
            new_node = ast.Assign(targets=[new_v], value=node)
            self.functionDef.body.insert(self.counter, new_node)
            self.counter += 1
            return new_v
        elif isinstance(node, ast.Name):
            return ast.Name(id=self.var_table[node.id], ctx=ast.Load())
        else:
            return None  # undefined

    def visit_BinOp(self, node):
        # Not directly visited for A @ B + C as it is contained in Return!

        if isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name):
            # v_c = A + B
            return self.Generate_BinOp_SAC(node)
        else:
            return self.Generate_SAC_recursively(node)  # recursion

    def Generate_BinOp_SAC(self, binop_node):
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
        self.functionDef.body.insert(self.counter, new_node)  # insert SAC

        print("Created {}".format(new_node.targets[0].id))
        self.counter += 1
        return new_v

    def visit_FunctionDef(self, node):
        self.functionDef = deepcopy(node)  # copy FuncDef to modify it
        self.functionDef.name = node.name + "_SAC"
        print("Generating {} ...".format(str(self.functionDef.name)))

        # clear the function body
        self.functionDef.body = []

        # recursively visit original function body AST (and update functionDef in the process)
        for child in ast.iter_child_nodes(node):
            self.visit(child)

        print("Final var dict: {}".format(str(self.var_table)))
        return self.functionDef  # return modified FunctionDef

    def visit_arguments(self, arguments):
        """Generates SAC for each function argument"""

        for a in arguments.args:
            new_v = ast.Name(id="v" + str(self.counter), ctx=ast.Store())
            old_v = ast.Name(id=a.arg, ctx=ast.Load())
            new_node = ast.Assign(targets=[new_v], value=old_v)
            self.functionDef.body.insert(self.counter, new_node)
            self.var_table[a.arg] = new_v.id
            print("Created {}".format(new_v.id))
            self.counter += 1

    def Generate_Return_SAC(self, return_node):
        new_return = ast.Return(
            value=ast.Name(id="v" + str(self.counter - 1), ctx=ast.Load())
        )
        self.functionDef.body.append(new_return)  # return last assignment on last line

    def visit_Return(self, node):
        self.Generate_SAC_recursively(node)
        return None  # not used

    def visit_Assign(self, node: ast.Assign) -> None:
        self.Generate_SAC_recursively(node)
        return None

    def visit_Constant(self, node: ast.Constant) -> ast.Name:
        new_v = ast.Name(id="v" + str(self.counter), ctx=ast.Load())
        new_node = ast.Assign(targets=[new_v], value=node)
        self.functionDef.body.insert(self.counter, new_node)
        self.counter += 1
        return new_v

    def visit_Call(self, node: ast.Call) -> ast.Name:
        return self.Generate_SAC_recursively(node)
