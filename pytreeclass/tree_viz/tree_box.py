from __future__ import annotations

import pytreeclass as pytc
from pytreeclass.tree_viz.box_drawing import _layer_box, _vbox
from pytreeclass.tree_viz.node_pprint import _format_node_repr
from pytreeclass.tree_viz.utils import _sequential_tree_shape_eval


def tree_box(tree, array=None):
    """Return subclass relations in a boxed style.

    Example:
        @pytc.treeclass
        class L0:
            a: int = 1
            b: int = 2

        @pytc.treeclass
        class L1:
            c: L0 = L0()
            d: int = 4

        >>> print(L1().tree_box())
        ┌─────────────────────────┐
        │L1[Parent]               │
        ├─────────────────────────┤
        │┌───────┬────────┬──────┐│
        ││       │ Input  │ None ││
        ││ L0[c] │────────┼──────┤│
        ││       │ Output │ None ││
        │└───────┴────────┴──────┘│
        │┌───┐                    │
        ││d=4│                    │
        │└───┘                    │
        └─────────────────────────┘
    """

    def recurse(tree, parent_name):

        nonlocal shapes

        if pytc.is_treeclass_leaf(tree):
            frozen_stmt = "(Frozen)" if pytc.is_treeclass_frozen(tree) else ""
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

            for field_item in pytc.fields(tree):
                cur_node = getattr(tree, field_item.name)
                level_nodes += (
                    [f"{recurse(cur_node,field_item.name)}"]
                    if pytc.is_treeclass(cur_node)
                    else [_vbox(f"{field_item.name}={_format_node_repr(cur_node,0)}")]
                )

            return _vbox(
                f"{tree.__class__.__name__}[{parent_name}]", "\n".join(level_nodes)
            )

    shapes = _sequential_tree_shape_eval(tree, array) if array is not None else None
    return recurse(tree, "Parent")
