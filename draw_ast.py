import ast
import inspect
import graphviz as viz
import ast_traversal


def draw_AST(function):
    """Draws the given AST in a Jupyter Notebook Cell.
        TODO: add edge labels
    Args:
        tree (_type_): _description_
    """
    vis = ast_traversal.RecordGraphVisitor()
    code = inspect.getsource(function) if callable(function) else function
    tree = ast.parse(code)
    vis.visit(tree)
    return vis.graph
