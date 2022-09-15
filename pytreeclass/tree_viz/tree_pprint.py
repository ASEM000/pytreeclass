from __future__ import annotations

from dataclasses import Field, field
from typing import Any

import pytreeclass._src as src
from pytreeclass._src.dispatch import dispatch
from pytreeclass._src.tree_util import (
    _tree_fields,
    is_frozen_field,
    is_nondiff_field,
    is_treeclass,
    is_treeclass_frozen,
    is_treeclass_nondiff,
)
from pytreeclass.tree_viz.node_pprint import (
    _format_node_diagram,
    _format_node_repr,
    _format_node_str,
    _format_width,
)


def _marker(field_item: Field, node_item: Any, default: str = "") -> str:
    """return the suitable marker given the field and node item

    Args:
        field_item (Field): field item of the pytree node
        node_item (Any): node item
        default (str, optional): default marker. Defaults to "".

    Returns:
        str: marker character.
    """
    # for now, we only have two markers '*' for non-diff and '#' for frozen
    if is_nondiff_field(field_item) or is_treeclass_nondiff(node_item):
        return "*"
    elif is_frozen_field(field_item) or is_treeclass_frozen(node_item):
        return "#"
    else:
        return default


def tree_repr(tree, width: int = 60) -> str:
    """Prertty print `treeclass_leaves`

    Returns:
        str: indented tree leaves.
    """

    @dispatch(argnum="node_item")
    def recurse_field(
        field_item: Field, node_item: Any, depth: int, is_last_field: bool
    ):
        """format non-treeclass field"""
        nonlocal FMT

        if field_item.repr:
            mark = _marker(field_item, node_item)
            FMT += "\n" + "\t" * depth
            FMT += f"{mark}{field_item.name}"
            FMT += "="
            FMT += f"{(_format_node_repr(node_item,depth))}"
            FMT += "" if is_last_field else ","
        recurse(node_item, depth)

    @recurse_field.register(src.tree_base._treeBase)
    def _(field_item: Field, node_item: Any, depth: int, is_last_field: bool):
        """format treeclass field"""
        nonlocal FMT

        if field_item.repr:
            mark = _marker(field_item, node_item)
            FMT += "\n" + "\t" * depth
            layer_class_name = f"{node_item.__class__.__name__}"
            FMT += f"{mark}{field_item.name}"
            FMT += f"={layer_class_name}" + "("
            start_cursor = len(FMT)  # capture children repr
            recurse(tree=node_item, depth=depth + 1)
            temp = FMT[:start_cursor]
            temp += _format_width(FMT[start_cursor:] + "\n" + "\t" * (depth) + ")")
            temp += "" if is_last_field else ","
            FMT = temp

    def recurse(tree, depth):
        if not is_treeclass(tree):
            return

        nonlocal FMT

        leaves_count = len(_tree_fields(tree))
        for i, fi in enumerate(_tree_fields(tree).values()):
            recurse_field(
                field_item=fi,
                node_item=getattr(tree, fi.name),
                depth=depth,
                is_last_field=True if i == (leaves_count - 1) else False,
            )

    FMT = ""
    recurse(tree=tree, depth=1)
    FMT = f"{(tree.__class__.__name__)}(" + _format_width(FMT + "\n)", width)

    return FMT.expandtabs(2)


def tree_str(tree, width: int = 40) -> str:
    """Prertty print `treeclass_leaves`

    Returns:
        str: indented tree leaves.
    """

    @dispatch(argnum="node_item")
    def recurse_field(
        field_item: Field, node_item: Any, depth: int, is_last_field: bool
    ):
        """format non-treeclass field"""
        nonlocal FMT

        if field_item.repr:
            mark = _marker(field_item, node_item)
            FMT += "\n" + "\t" * depth
            FMT += f"{mark}{field_item.name}"
            FMT += "="

            if "\n" in f"{node_item!s}":
                FMT += "\n" + "\t" * (depth + 1)
                FMT += f"{(_format_node_str(node_item,depth+1))}"
            else:
                FMT += f"{(_format_node_str(node_item,depth))}"

            FMT += "" if is_last_field else ","

        recurse(node_item, depth)

    @recurse_field.register(src.tree_base._treeBase)
    def _(field_item: Field, node_item: Any, depth: int, is_last_field: bool):
        """format treeclass field"""
        nonlocal FMT

        if field_item.repr:
            # mark a module static if all its fields are static
            mark = _marker(field_item, node_item)

            FMT += "\n" + "\t" * depth
            layer_class_name = f"{node_item.__class__.__name__}"

            FMT += f"{mark}{field_item.name}"
            FMT += f"={layer_class_name}" + "("
            start_cursor = len(FMT)  # capture children repr

            recurse(node_item, depth=depth + 1)

            temp = FMT[:start_cursor]
            temp += _format_width(FMT[start_cursor:] + "\n" + "\t" * (depth) + ")")
            temp += "" if is_last_field else ","
            FMT = temp

    def recurse(tree, depth):
        if not is_treeclass(tree):
            return
        nonlocal FMT
        leaves_count = len(_tree_fields(tree))
        for i, fi in enumerate(_tree_fields(tree).values()):
            recurse_field(
                field_item=fi,
                node_item=getattr(tree, fi.name),
                depth=depth,
                is_last_field=True if i == (leaves_count - 1) else False,
            )

    FMT = ""
    recurse(tree, depth=1)
    FMT = f"{(tree.__class__.__name__)}(" + _format_width(FMT + "\n)", width)

    return FMT.expandtabs(2)


def _tree_diagram(tree):
    """
    === Explanation
        pretty print treeclass tree with tree structure diagram

    === Args
        tree : boolean to create tree-structure
    """

    @dispatch(argnum=1)
    def recurse_field(field_item, node_item, parent_level_count, node_index):
        nonlocal FMT

        if field_item.repr:
            mark = _marker(field_item, node_item, default="─")
            is_last_field = node_index <= 1

            FMT += "\n"
            FMT += "".join(
                [(("│" if lvl > 1 else "") + "\t") for lvl in parent_level_count]
            )

            FMT += f"└{mark}─ " if is_last_field else f"├{mark}─ "
            FMT += f"{field_item.name}"
            FMT += f"={_format_node_diagram(node_item)}"

        recurse(node_item, parent_level_count + [1])

    @recurse_field.register(list)
    @recurse_field.register(tuple)
    def _(field_item, node_item, parent_level_count, node_index):
        nonlocal FMT

        if field_item.repr:
            recurse_field(
                field_item,
                node_item.__class__,
                parent_level_count,
                node_index,
            )

            for i, layer in enumerate(node_item):
                new_field = field(
                    metadata={
                        "static": is_nondiff_field(field_item),
                        "frozen": is_frozen_field(field_item),
                    }
                )

                object.__setattr__(new_field, "name", f"{field_item.name}_{i}")
                object.__setattr__(new_field, "type", type(layer))

                recurse_field(
                    new_field,
                    layer,
                    parent_level_count + [node_index],
                    len(node_item) - i,
                )

        recurse(node_item, parent_level_count)

    @recurse_field.register(src.tree_base._treeBase)
    def _(field_item, node_item, parent_level_count, node_index):
        nonlocal FMT

        if field_item.repr:
            mark = _marker(field_item, node_item, default="─")
            layer_class_name = node_item.__class__.__name__
            is_last_field = node_index == 1
            FMT += "\n" + "".join(
                [(("│" if lvl > 1 else "") + "\t") for lvl in parent_level_count]
            )
            FMT += f"└{mark}─ " if is_last_field else f"├{mark}─ "
            FMT += f"{field_item.name}"
            FMT += f"={layer_class_name}"

            recurse(node_item, parent_level_count + [node_index])

    def recurse(tree, parent_level_count):
        if not is_treeclass(tree):
            return

        nonlocal FMT

        leaves_count = len(_tree_fields(tree))

        for i, fi in enumerate(_tree_fields(tree).values()):
            recurse_field(
                fi,
                getattr(tree, fi.name),
                parent_level_count,
                leaves_count - i,
            )

        FMT += "\t"

    FMT = f"{(tree.__class__.__name__)}"

    recurse(tree, [1])

    return FMT.expandtabs(4)


def tree_diagram(tree, link=False):
    string = _tree_diagram(tree)

    # if link:
    #     string = string.replace("   ", "&emsp;&emsp;")
    #     string = "".join(map(_mermaid_table_row, string.split("\n")))
    #     string = "flowchart TD\n" + "A[" + '"' + _mermaid_table(string) + '"' + "]"
    #     return _generate_mermaid_link(string)

    # else:
    return string
