from __future__ import annotations

from dataclasses import field

import pytreeclass._src as src
from pytreeclass._src.dispatch import dispatch
from pytreeclass._src.tree_util import _tree_fields, is_treeclass, is_treeclass_frozen
from pytreeclass.tree_viz.node_pprint import (
    _format_node_diagram,
    _format_node_repr,
    _format_node_str,
    _format_width,
)


def tree_repr(tree, width: int = 60) -> str:
    """Prertty print `treeclass_leaves`

    Returns:
        str: indented tree leaves.
    """

    @dispatch(argnum=1)
    def recurse_field(field_item, node_item, depth, is_frozen, is_last_field):
        """format non-treeclass field"""
        nonlocal FMT

        if field_item.repr:
            is_static = field_item.metadata.get("static", False)
            mark = "*" if is_static else ("#" if is_frozen else "")
            FMT += "\n" + "\t" * depth
            FMT += f"{mark}{field_item.name}"
            FMT += "="
            FMT += f"{(_format_node_repr(node_item,depth))}"
            FMT += "" if is_last_field else ","
        recurse(node_item, depth, is_frozen)

    @recurse_field.register(src.tree_base._treeBase)
    def _(field_item, node_item, depth, is_frozen, is_last_field):
        """format treeclass field"""
        nonlocal FMT

        if field_item.repr:
            is_frozen = is_treeclass_frozen(node_item)
            is_static = field_item.metadata.get("static", False)
            mark = "*" if is_static else ("#" if is_frozen else "")

            FMT += "\n" + "\t" * depth
            layer_class_name = f"{node_item.__class__.__name__}"

            FMT += f"{mark}{field_item.name}"
            FMT += f"={layer_class_name}" + "("
            start_cursor = len(FMT)  # capture children repr

            recurse(
                node_item, depth=depth + 1, is_frozen=is_treeclass_frozen(node_item)
            )

            FMT = FMT[:start_cursor] + _format_width(
                FMT[start_cursor:] + "\n" + "\t" * (depth) + ")"
            )
            FMT += "" if is_last_field else ","

    def recurse(tree, depth, is_frozen):
        if not is_treeclass(tree):
            return

        nonlocal FMT

        leaves_count = len(_tree_fields(tree))
        for i, fi in enumerate(_tree_fields(tree).values()):
            recurse_field(
                fi,
                getattr(tree, fi.name),
                depth,
                is_frozen,
                True if i == (leaves_count - 1) else False,
            )

    FMT = ""
    recurse(tree, depth=1, is_frozen=is_treeclass_frozen(tree))
    FMT = f"{(tree.__class__.__name__)}(" + _format_width(FMT + "\n)", width)

    return FMT.expandtabs(2)


def tree_str(tree, width: int = 40) -> str:
    """Prertty print `treeclass_leaves`

    Returns:
        str: indented tree leaves.
    """

    @dispatch(argnum=1)
    def recurse_field(field_item, node_item, depth, is_frozen, is_last_field):
        """format non-treeclass field"""
        nonlocal FMT

        if field_item.repr:
            is_static = field_item.metadata.get("static", False)
            mark = "*" if is_static else ("#" if is_frozen else "")

            FMT += "\n" + "\t" * depth
            FMT += f"{mark}{field_item.name}"
            FMT += "="

            if "\n" in f"{node_item!s}":
                FMT += "\n" + "\t" * (depth + 1)
                FMT += f"{(_format_node_str(node_item,depth+1))}"
            else:
                FMT += f"{(_format_node_str(node_item,depth))}"

            FMT += "" if is_last_field else ","

        recurse(node_item, depth, is_frozen)

    @recurse_field.register(src.tree_base._treeBase)
    def _(field_item, node_item, depth, is_frozen, is_last_field):
        """format treeclass field"""
        nonlocal FMT

        if field_item.repr:
            is_frozen = is_treeclass_frozen(node_item)
            is_static = field_item.metadata.get("static", False)
            mark = "*" if is_static else ("#" if is_frozen else "")

            FMT += "\n" + "\t" * depth
            layer_class_name = f"{node_item.__class__.__name__}"

            FMT += f"{mark}{field_item.name}"
            FMT += f"={layer_class_name}" + "("
            start_cursor = len(FMT)  # capture children repr

            recurse(
                node_item, depth=depth + 1, is_frozen=is_treeclass_frozen(node_item)
            )

            FMT = FMT[:start_cursor] + _format_width(
                FMT[start_cursor:] + "\n" + "\t" * (depth) + ")"
            )
            FMT += "" if is_last_field else ","

    def recurse(tree, depth, is_frozen):
        if not is_treeclass(tree):
            return
        nonlocal FMT
        leaves_count = len(_tree_fields(tree))
        for i, fi in enumerate(_tree_fields(tree).values()):
            recurse_field(
                fi,
                getattr(tree, fi.name),
                depth,
                is_frozen,
                True if i == (leaves_count - 1) else False,
            )

    FMT = ""
    recurse(tree, depth=1, is_frozen=is_treeclass_frozen(tree))
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
    def recurse_field(field_item, node_item, is_frozen, parent_level_count, node_index):
        nonlocal FMT

        if field_item.repr:
            is_static = field_item.metadata.get("static", False)
            mark = "*" if is_static else ("#" if is_frozen else "─")
            is_last_field = node_index <= 1

            FMT += "\n"
            FMT += "".join(
                [(("│" if lvl > 1 else "") + "\t") for lvl in parent_level_count]
            )

            FMT += f"└{mark}─ " if is_last_field else f"├{mark}─ "
            FMT += f"{field_item.name}"
            FMT += f"={_format_node_diagram(node_item)}"

        recurse(node_item, parent_level_count + [1], is_frozen)

    @recurse_field.register(list)
    @recurse_field.register(tuple)
    def _(field_item, node_item, is_frozen, parent_level_count, node_index):
        nonlocal FMT

        if field_item.repr:
            recurse_field(
                field_item,
                node_item.__class__,
                is_frozen,
                parent_level_count,
                node_index,
            )

            for i, layer in enumerate(node_item):
                new_field = field()
                object.__setattr__(new_field, "name", f"{field_item.name}_{i}")
                object.__setattr__(new_field, "type", type(layer))

                recurse_field(
                    new_field,
                    layer,
                    is_frozen,
                    parent_level_count + [node_index],
                    len(node_item) - i,
                )

        recurse(node_item, parent_level_count, is_frozen)

    @recurse_field.register(src.tree_base._treeBase)
    def _(field_item, node_item, is_frozen, parent_level_count, node_index):
        nonlocal FMT

        if field_item.repr:
            is_frozen = is_treeclass_frozen(node_item)
            is_static = field_item.metadata.get("static", False)
            mark = "*" if is_static else ("#" if is_frozen else "─")
            layer_class_name = node_item.__class__.__name__

            is_last_field = node_index == 1

            FMT += "\n" + "".join(
                [(("│" if lvl > 1 else "") + "\t") for lvl in parent_level_count]
            )

            FMT += f"└{mark}─ " if is_last_field else f"├{mark}─ "
            FMT += f"{field_item.name}"
            FMT += f"={layer_class_name}"

            recurse(node_item, parent_level_count + [node_index], is_frozen)

    def recurse(tree, parent_level_count, is_frozen):
        if not is_treeclass(tree):
            return

        nonlocal FMT

        leaves_count = len(_tree_fields(tree))

        for i, fi in enumerate(_tree_fields(tree).values()):
            recurse_field(
                fi,
                getattr(tree, fi.name),
                is_frozen,
                parent_level_count,
                leaves_count - i,
            )

        FMT += "\t"

    FMT = f"{(tree.__class__.__name__)}"

    recurse(tree, [1], is_treeclass_frozen(tree))

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
