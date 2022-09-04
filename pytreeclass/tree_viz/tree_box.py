from __future__ import annotations

from pytreeclass.src.tree_util import (
    is_treeclass,
    is_treeclass_leaf,
    sequential_tree_shape_eval,
)
from pytreeclass.tree_viz.box_drawing import _layer_box, _vbox
from pytreeclass.tree_viz.node_pprint import _format_node_repr


def tree_box(tree, array=None):
    """
    === plot tree classes
    """

    def recurse(tree, parent_name):

        nonlocal shapes

        if is_treeclass_leaf(tree):
            frozen_stmt = "(Frozen)" if tree.frozen else ""
            box = _layer_box(
                f"{tree.__class__.__name__}[{parent_name}]{frozen_stmt}",
                _format_node_repr(shapes[0], 0) if array is not None else None,
                _format_node_repr(shapes[1], 0) if array is not None else None,
            )

            if shapes is not None:
                shapes.pop(0)
            return box

        else:
            level_nodes = []

            for fi in tree.__pytree_fields__.values():
                cur_node = tree.__dict__[fi.name]

                if is_treeclass(cur_node):
                    level_nodes += [f"{recurse(cur_node,fi.name)}"]

                else:
                    level_nodes += [_vbox(f"{fi.name}={_format_node_repr(cur_node,0)}")]

            return _vbox(
                f"{tree.__class__.__name__}[{parent_name}]", "\n".join(level_nodes)
            )

    shapes = sequential_tree_shape_eval(tree, array) if array is not None else None
    return recurse(tree, "Parent")
