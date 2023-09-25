# some convenience methods to traverse the AST

import ast
import graphviz as viz
import inspect
import subprocess
import pathlib
from typing import Callable, Union


class VerboseRecursiveVisitor(ast.NodeVisitor):
    """Recursive node visitor (verbose output)"""

    def __init__(self):
        self.n = 0
        self.cur_parent = None
        self.indent_c = 0  # TODO

    def recursive(func):
        """decorator to make visitor work recursive"""

        def wrapper(self, node):
            self.n += 1
            func(self, node)  # visit node itself
            print(
                'Node "{node}" (#{i} overall) found {N} children.'.format(
                    node=type(node).__name__,
                    i=self.n,
                    N=sum(1 for _ in ast.iter_child_nodes(node)),
                )
            )
            if self.cur_parent is not None:
                print("Parent node = {p}".format(p=self.cur_parent))

            self.cur_parent = node
            for child in ast.iter_child_nodes(node):
                self.visit(child)

        return wrapper

    def print_attributes(self, node):
        print('Attributes of "{n}":'.format(n=type(node).__name__))
        for i in ast.iter_fields(node):
            print(i)

    @recursive
    def visit_Assign(self, node):
        print(type(node).__name__)
        self.print_attributes(node)

    @recursive
    def visit_BinOp(self, node):
        print(type(node).__name__)
        self.print_attributes(node)

    @recursive
    def visit_Call(self, node):
        print(type(node).__name__)
        self.print_attributes(node)

    @recursive
    def visit_FunctionDef(self, node):
        print(type(node).__name__)
        self.print_attributes(node)

    @recursive
    def visit_Return(self, node):
        print(type(node).__name__)
        self.print_attributes(node)

    @recursive
    def visit_MatMult(self, node):
        print(type(node).__name__)
        self.print_attributes(node)

    @recursive
    def visit_Name(self, node):
        print(type(node).__name__)
        self.print_attributes(node)

    @recursive
    def visit_Module(self, node):
        """visit a Module node and the visits recursively"""
        pass

    def generic_visit(self, node):
        pass


class RecordGraphVisitor(ast.NodeVisitor):
    """Recursive node visitor"""

    def __init__(self):
        self.graph = viz.Digraph(graph_attr={"size": "9,5"})
        self.print_attr = ["name", "id", "arg", "op", "attr"]
        self.parse_attr = [
            "name",
            "id",
            "arg",
            "args",
            "body",
            "value",
            "left",
            "right",
            "op",
            "attr",
        ]
        # global counters for unique node ids (how can this be done S-Attributed?)
        # there can only be one Module
        self.FunctionDef_c = 0
        self.Name_c = 0
        self.Assign_c = 0
        self.AugAssign_c = 0
        self.BinOp_c = 0
        self.Call_c = 0
        self.Attr_c = 0
        self.Return_c = 0

        self.cur_parent = None

    def recursive(func):
        """decorator to make visitor work recursively"""

        def wrapper(self, node):
            oldparent = self.cur_parent
            func(self, node)  # visit node itself

            for child in ast.iter_child_nodes(node):
                self.visit(child)  # visit all children
            self.cur_parent = oldparent

        return wrapper

    def add_node(self, ast_node, id):
        name = str(type(ast_node).__name__)
        text = name + "\n"
        for attr in ast.iter_fields(ast_node):  # tuple
            if attr[0] in self.print_attr:
                if isinstance(ast_node, ast.BinOp):  # pretty print
                    text = (
                        text
                        + str(attr[0])
                        + ": "
                        + str(ast_node.op.__class__.__name__)
                        + "\n"
                    )
                elif isinstance(ast_node, ast.AugAssign):
                    continue
                else:  # default print
                    text = text + str(attr[0]) + ": " + str(attr[1]) + "\n"

        unique_id = name + str(id or "")
        self.graph.node(unique_id, label=text)  # add node to graph

        if id is not None:  # reserved for root Module
            self.graph.edge(self.cur_parent, unique_id)
            self.cur_parent = unique_id  # remember for child nodes (broken)

    # --- recursive node visitors (somewhat hierachical order) ---
    # visit functions can return a value which is forwarded!

    @recursive
    def visit_Module(self, node):
        self.add_node(node, None)
        self.cur_parent = "Module"

    @recursive
    def visit_FunctionDef(self, node):
        self.add_node(node, self.FunctionDef_c)
        self.FunctionDef_c += 1

    @recursive
    def visit_Name(self, node):
        self.add_node(node, self.Name_c)
        self.Name_c += 1

    @recursive
    def visit_Assign(self, node):
        self.add_node(node, self.Assign_c)
        self.Assign_c += 1

    @recursive
    def visit_AugAssign(self, node):
        self.add_node(node, self.AugAssign_c)
        self.AugAssign_c += 1

    @recursive
    def visit_BinOp(self, node):
        self.add_node(node, self.BinOp_c)
        self.BinOp_c += 1

    @recursive
    def visit_Call(self, node):
        self.add_node(node, self.Call_c)
        self.Call_c += 1

    @recursive
    def visit_Attribute(self, node):
        self.add_node(node, self.Attr_c)
        self.Attr_c += 1

    @recursive
    def visit_Return(self, node):
        self.add_node(node, self.Return_c)
        self.Return_c += 1

    def generic_visit(self, node):
        pass


def print_AST(function: Union[str, Callable]):
    """Prints the AST of the given function.

    Args:
        function (str or function): the function object or its code
    """
    tree = (
        ast.parse(inspect.getsource(function))
        if isinstance(function, Callable)
        else ast.parse(function)
    )
    v = VerboseRecursiveVisitor()
    v.visit(tree)


def draw_AST(function: Union[str, Callable], export=False):
    """Draws the given AST in a Jupyter Notebook Cell.
    TODO: add edge labels
    Args:
        function (str or function): the function object or its code
    """
    vis = RecordGraphVisitor()
    tree = (
        ast.parse(inspect.getsource(function))
        if isinstance(function, Callable)
        else ast.parse(function)
    )
    vis.visit(tree)

    if export:
        path = pathlib.Path("./graph.dot")
        with open(path, "w") as out:
            out.write(vis.graph.source)
        subprocess.run(["dot", "-Tsvg", "-o", "graph.svg", "graph.dot"])

    return vis.graph
