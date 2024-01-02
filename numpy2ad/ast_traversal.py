import ast
import graphviz as viz
import inspect
import subprocess
import pathlib
from typing import Callable, Union


# TODO: make this optional at install time as it introduces Graphviz CLI dependency (which we can't package)
class RecordGraphVisitor(ast.NodeVisitor):
    """Recursive node visitor"""

    def __init__(self):
        self.graph = viz.Digraph()
        self.print_attr = ["name", "id", "arg", "op", "attr"]

        # global counters for unique node ids
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
            oldparent = self.cur_parent  # remember parent node
            func(self, node)  # visit node itself

            for child in ast.iter_child_nodes(node):
                self.visit(child)  # visit all children
            self.cur_parent = oldparent  # restore parent node

        return wrapper

    def add_node(self, ast_node, id: str) -> None:
        name = str(type(ast_node).__name__)
        text = name + "\n"
        for attr in ast.iter_fields(ast_node):  # tuple
            if attr[0] in self.print_attr:
                # TODO: extract these printing rules to node visitors
                if isinstance(ast_node, ast.BinOp):  # pretty print
                    text = text + str(attr[0]) + ": " + str(get_name(ast_node.op))
                elif isinstance(ast_node, ast.AugAssign):
                    text = text + str(attr[0]) + ": " + str(get_name(ast_node.op))
                else:  # default print
                    text = text + str(attr[0]) + ": " + str(attr[1])

        unique_id = name + str(id or "")
        self.graph.node(unique_id, label=text)  # add node to graph

        if id is not None:  # reserved for root Module
            self.graph.edge(self.cur_parent, unique_id)
            self.cur_parent = unique_id  # remember for child nodes

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


def get_name(obj) -> str:
    return str(obj.__class__.__name__)


class RecordMinimalGraphVisitor(ast.NodeVisitor):
    """Recursive node visitor with minimal node text"""

    def __init__(self):
        self.graph = viz.Digraph()

        # minimal nodes
        self.minimal_ops = {
            "Add": "+",
            "UAdd": "+",
            "Sub": "-",
            "USub": "-",
            "MatMult": "@",
            "Assign": "=",
        }

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
        self.attr_buff = ""

    def recursive(func):
        """decorator to make visitor work recursively and to update self.cur_parent"""

        def wrapper(self, node):
            oldparent = self.cur_parent
            self.cur_parent = func(self, node)  # visit node itself

            for child in ast.iter_child_nodes(node):
                self.visit(child)  # visit all children (if any)
            self.cur_parent = oldparent  # restore old parent

        return wrapper

    def recursive_attr(visit):
        """decorator to make ast.Attribute visitor work recursively"""

        def wrapper(self, node: ast.Attribute):
            """builds one node string"""

            # prepend to buffer
            self.attr_buff = visit(node.attr) + self.attr_buff

            # end recursion

            self.attr_buff = ""  # reset

        return wrapper

    def add_node(self, id: str, text: str):
        self.graph.node(id, label=text)  # add node to graph
        if id != "root":
            self.graph.edge(self.cur_parent, id)

    # --- recursive node visitors ---
    # visit methods must return node id as string for edges to children!

    @recursive
    def visit_Module(self, node: ast.Module) -> str:
        graph_id = "root"
        self.add_node(graph_id, "Module")
        return graph_id

    @recursive
    def visit_FunctionDef(self, node: ast.FunctionDef) -> str:
        graph_id = f"FunctionDef{self.FunctionDef_c}"
        self.add_node(graph_id, node.name)
        self.FunctionDef_c += 1
        return graph_id

    @recursive
    def visit_Name(self, node: ast.Name) -> str:
        graph_id = f"Name{self.Name_c}"
        self.add_node(graph_id, str(node.id))
        self.Name_c += 1
        return graph_id

    @recursive
    def visit_Assign(self, node: ast.Assign) -> str:
        graph_id = f"Assign{self.Assign_c}"
        self.add_node(graph_id, self.minimal_ops[get_name(node)])
        self.Assign_c += 1
        return graph_id

    @recursive
    def visit_AugAssign(self, node: ast.AugAssign) -> str:
        graph_id = f"AugAssign{self.AugAssign_c}"
        self.add_node(graph_id, f"{self.minimal_ops[get_name(node.op)]}=")
        self.AugAssign_c += 1
        return graph_id

    @recursive
    def visit_BinOp(self, node) -> str:
        graph_id = f"BinOp{self.BinOp_c}"
        self.add_node(graph_id, self.minimal_ops[get_name(node.op)])
        self.BinOp_c += 1
        return graph_id

    @recursive
    def visit_Call(self, node) -> str:
        graph_id = f"Call{self.Call_c}"
        # node.func is a Name or Attribute
        self.add_node(graph_id, "Call")
        self.Call_c += 1
        return graph_id

    def visit_Attribute(self, node: ast.Attribute) -> str:
        if self.attr_buff == "":
            self.attr_buff = f".{node.attr}"
            self.visit_Attribute(node.value)

            # insert attribute node into graph
            graph_id = f"Attribute{self.Attr_c}"
            self.add_node(graph_id, self.attr_buff)
            self.Attr_c += 1
            self.attr_buff = ""
            return graph_id
        else:  # recursion
            if isinstance(node, ast.Name):
                self.attr_buff = f"{node.id}{self.attr_buff}"
                return  # end of recursion
            elif isinstance(node, ast.Attribute):
                self.attr_buff = f".{node.attr}{self.attr_buff}"
                self.visit_Attribute(node.value)
            else:
                raise NotImplementedError(f"unexpected {node.value}")

    @recursive
    def visit_Return(self, node) -> str:
        graph_id = f"Return{self.Return_c}"
        self.add_node(graph_id, "Return")
        self.Return_c += 1
        return graph_id

    def generic_visit(self, node):
        pass


def draw_AST(function: Union[str, Callable], export=False, minimal=False):
    """Draws the given AST in a Jupyter Notebook Cell. Requires `graphviz` and `dot` to be installed.
    Args:
        function (str or function): Python expression or function object
        export (bool): exports the graph as `graph.pdf`
        minimal (bool): generates graph with "minimal" node text
    """
    vis = RecordGraphVisitor() if not minimal else RecordMinimalGraphVisitor()
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
        subprocess.run(["dot", "-Tpdf", "-o", "graph.pdf", "graph.dot"])

    return vis.graph
