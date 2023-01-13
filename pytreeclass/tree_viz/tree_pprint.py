from __future__ import annotations

import dataclasses as dc
from typing import Any

from pytreeclass.tree_viz.node_pprint import (
    _format_node_repr,
    _format_node_str,
    _format_width,
)
from pytreeclass.tree_viz.utils import _marker

PyTree = Any


def _tree_pprint(tree, width: int = 80, kind="repr") -> str:
    """Prertty print `treeclass_leaves`"""

    def recurse(tree: PyTree, depth: int):
        if not dc.is_dataclass(tree):
            return

        nonlocal FMT

        leaves_count = len(dc.fields(tree))
        for i, field_item in enumerate(dc.fields(tree)):

            if not field_item.repr:
                # skip fields with `repr=False`
                continue

            node_item = getattr(tree, field_item.name)
            mark = _marker(field_item, node_item)
            endl = "" if i == (leaves_count - 1) else ","
            FMT += "\n" + "\t" * depth

            if dc.is_dataclass(node_item):
                FMT += f"{mark}{field_item.name}={node_item.__class__.__name__}("
                cursor = len(FMT)  # capture children repr
                recurse(tree=node_item, depth=depth + 1)
                _FMT = FMT
                FMT = _FMT[:cursor]
                FMT += _format_width(_FMT[cursor:] + "\n" + "\t" * (depth) + ")")
                FMT += endl

            else:
                FMT += f"{mark}{field_item.name}="

                if kind == "repr":
                    FMT += f"{(_format_node_repr(node_item,depth))}"

                elif kind == "str":
                    if "\n" in f"{node_item!s}":
                        # in case of multiline string then indent all lines
                        FMT += "\n" + "\t" * (depth + 1)
                        FMT += f"{(_format_node_str(node_item,depth+1))}"
                    else:
                        FMT += f"{(_format_node_str(node_item,depth))}"

                FMT += endl
                recurse(tree=node_item, depth=depth)

    FMT = ""
    recurse(tree=tree, depth=1)
    FMT = f"{(tree.__class__.__name__)}(" + _format_width(FMT + "\n)", width)
    return FMT.expandtabs(2)


def tree_repr(tree, width: int = 80) -> str:
    """Prertty print `treeclass_leaves`"""
    return _tree_pprint(tree, width, kind="repr")


def tree_str(tree, width: int = 80) -> str:
    """Prertty print `treeclass_leaves`"""
    return _tree_pprint(tree, width, kind="str")
