from __future__ import annotations

import dataclasses
from typing import Any

import pytreeclass._src.dataclass_util as dcu
from pytreeclass.tree_viz.node_pprint import (
    _format_node_diagram,
    _format_node_repr,
    _format_node_str,
    _format_width,
)

PyTree = Any


def _marker(field_item: dataclasses.Field, node_item: Any, default: str = "") -> str:
    """return the suitable marker given the field and node item"""
    # '*' for non-diff
    if dcu.is_field_nondiff(field_item) or dcu.is_dataclass_fields_nondiff(node_item):
        return "*"
    elif dcu.is_field_frozen(field_item) or dcu.is_dataclass_fields_frozen(node_item):
        return "#"
    return default


def tree_repr(tree, width: int = 80) -> str:
    """Prertty print `treeclass_leaves`"""

    def recurse(tree: PyTree, depth: int):
        if not dataclasses.is_dataclass(tree):
            return

        nonlocal FMT

        leaves_count = len(dataclasses.fields(tree))
        for i, field_item in enumerate(dataclasses.fields(tree)):

            if not field_item.repr:
                continue

            node_item = getattr(tree, field_item.name)
            mark = _marker(field_item, node_item)
            endl = "" if i == (leaves_count - 1) else ","
            FMT += "\n" + "\t" * depth

            if dataclasses.is_dataclass(node_item):
                FMT += f"{mark}{field_item.name}={node_item.__class__.__name__}("
                cursor = len(FMT)  # capture children repr
                recurse(tree=node_item, depth=depth + 1)
                FMT = FMT[:cursor] + _format_width(FMT[cursor:] + "\n" + "\t" * (depth) + ")") + endl  # fmt: skip

            else:
                FMT += f"{mark}{field_item.name}={(_format_node_repr(node_item,depth))}" + endl  # fmt: skip
                recurse(tree=node_item, depth=depth)

    FMT = ""
    recurse(tree=tree, depth=1)
    FMT = f"{(tree.__class__.__name__)}(" + _format_width(FMT + "\n)", width)

    return FMT.expandtabs(2)


def tree_str(tree, width: int = 80) -> str:
    """Prertty print `treeclass_leaves`"""

    def recurse(tree, depth):
        if not dataclasses.is_dataclass(tree):
            return

        nonlocal FMT

        leaves_count = len(dataclasses.fields(tree))

        for i, field_item in enumerate(dataclasses.fields(tree)):

            if not field_item.repr:
                continue

            node_item = getattr(tree, field_item.name)
            mark = _marker(field_item, node_item)
            endl = "" if i == (leaves_count - 1) else ","
            FMT += "\n" + "\t" * depth

            if dataclasses.is_dataclass(node_item):
                FMT += f"{mark}{field_item.name}={node_item.__class__.__name__}" + "("
                cursor = len(FMT)
                recurse(tree=node_item, depth=depth + 1)
                FMT = FMT[:cursor] + _format_width(FMT[cursor:] + "\n" + "\t" * (depth) + ")") + endl  # fmt: skip

            else:
                FMT += f"{mark}{field_item.name}="

                if "\n" in f"{node_item!s}":
                    FMT += "\n" + "\t" * (depth + 1) + f"{(_format_node_str(node_item,depth+1))}" + endl  # fmt: skip
                else:
                    FMT += f"{(_format_node_str(node_item,depth))}" + endl

                recurse(tree=node_item, depth=depth)

    FMT = ""
    recurse(tree, depth=1)
    FMT = f"{(tree.__class__.__name__)}(" + _format_width(FMT + "\n)", width)

    return FMT.expandtabs(2)


def tree_diagram(tree: PyTree) -> str:
    """Pretty print treeclass tree with tree structure diagram"""

    def recurse_field(field_item, node_item, parent_level_count, node_index):
        nonlocal FMT

        if not field_item.repr:
            return

        if dataclasses.is_dataclass(node_item):
            mark = _marker(field_item, node_item, default="─")
            layer_class_name = node_item.__class__.__name__
            is_last_field = node_index == 1
            FMT += "\n" + "".join([(("│" if lvl > 1 else "") + "\t") for lvl in parent_level_count])  # fmt: skip
            FMT += f"└{mark}─ " if is_last_field else f"├{mark}─ "
            FMT += f"{field_item.name}"
            FMT += f"={layer_class_name}"

            recurse(node_item, parent_level_count + [node_index])

        elif isinstance(node_item, (list, tuple)) and any(
            dataclasses.is_dataclass(leaf) for leaf in (node_item)
        ):
            # expand a contaner if any item  in the container `is treeclass`
            recurse_field(field_item, node_item.__class__, parent_level_count, node_index)  # fmt: skip

            for i, layer in enumerate(node_item):
                if dcu.is_field_nondiff(field_item):
                    new_field = dataclasses.field(metadata={"static": "nondiff"})
                else:
                    new_field = dataclasses.field()

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
            FMT += f"={_format_node_diagram(node_item)}"

            recurse(node_item, parent_level_count + [1])

    def recurse(tree, parent_level_count):
        if not dataclasses.is_dataclass(tree):
            return

        nonlocal FMT

        leaves_count = len(dataclasses.fields(tree))

        for i, fi in enumerate(dataclasses.fields(tree)):
            recurse_field(fi, getattr(tree, fi.name), parent_level_count, leaves_count - i)  # fmt: skip

        FMT += "\t"

    FMT = f"{(tree.__class__.__name__)}"

    recurse(tree, [1])

    return FMT.expandtabs(4)
