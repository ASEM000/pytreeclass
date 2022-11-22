from __future__ import annotations

import dataclasses as dc
from typing import Any

import pytreeclass._src.dataclass_util as dcu
from pytreeclass.tree_viz.node_pprint import _format_node_repr
from pytreeclass.tree_viz.utils import _marker

PyTree = Any


def tree_diagram(tree: PyTree) -> str:
    """Pretty print treeclass tree with tree structure diagram"""

    def recurse_field(field_item, node_item, parent_level_count, node_index):
        nonlocal FMT

        if not field_item.repr:
            return

        if dc.is_dataclass(node_item):
            mark = _marker(field_item, node_item, default="─")
            layer_class_name = node_item.__class__.__name__
            is_last_field = node_index == 1
            FMT += "\n" + "".join([(("│" if lvl > 1 else "") + "\t") for lvl in parent_level_count])  # fmt: skip
            FMT += f"└{mark}─ " if is_last_field else f"├{mark}─ "
            FMT += f"{field_item.name}"
            FMT += f"={layer_class_name}"

            recurse(node_item, parent_level_count + [node_index])

        elif isinstance(node_item, (list, tuple)) and any(
            dc.is_dataclass(leaf) for leaf in (node_item)
        ):
            # expand a contaner if any item  in the container `is treeclass`
            recurse_field(field_item, node_item.__class__, parent_level_count, node_index)  # fmt: skip

            for i, layer in enumerate(node_item):
                if dcu.is_field_nondiff(field_item):
                    new_field = dc.field(metadata={"static": "nondiff"})
                else:
                    new_field = dc.field()

                object.__setattr__(new_field, "name", f"{field_item.name}[{i}]")
                object.__setattr__(new_field, "type", type(layer))

                recurse_field(new_field, layer, parent_level_count + [node_index], len(node_item) - i)  # fmt: skip

            recurse(node_item, parent_level_count)

        else:
            mark = _marker(field_item, node_item, default="─")
            is_last_field = node_index <= 1

            FMT += "\n"
            FMT += "".join([(("│" if lvl > 1 else "") + "\t") for lvl in parent_level_count])  # fmt: skip
            FMT += f"└{mark}─ " if is_last_field else f"├{mark}─ "
            FMT += f"{field_item.name}"
            FMT += f"={_format_node_repr(node_item)}"

            recurse(node_item, parent_level_count + [1])

    def recurse(tree, parent_level_count):
        if not dc.is_dataclass(tree):
            return

        nonlocal FMT

        leaves_count = len(dc.fields(tree))

        for i, fi in enumerate(dc.fields(tree)):
            recurse_field(fi, getattr(tree, fi.name), parent_level_count, leaves_count - i)  # fmt: skip

        FMT += "\t"

    FMT = f"{(tree.__class__.__name__)}"

    recurse(tree, [1])

    return FMT.expandtabs(4)
