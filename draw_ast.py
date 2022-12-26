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
    tree = ast.parse(inspect.getsource(function))
    vis.visit(tree)
    return vis.graph


def export_AST(tree, filename):
    dot = viz.Digraph(name="AST", filename=filename)
