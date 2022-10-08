from __future__ import annotations

import ctypes
from dataclasses import Field
from typing import Any

import pytreeclass as pytc
import pytreeclass._src as src
from pytreeclass.tree_viz.node_pprint import _format_node_diagram
from pytreeclass.tree_viz.tree_export import _generate_mermaid_link
from pytreeclass.tree_viz.tree_summary import (
    _format_count,
    _format_size,
    _reduce_count_and_size,
)


def _marker(field_item: Field, node_item: Any, default: str = "--") -> str:
    """return the suitable marker given the field and node item

    Args:
        field_item (Field): field item of the pytree node
        node_item (Any): node item
        default (str, optional): default marker. Defaults to "".

    Returns:
        str: marker character.
    """
    # for now, we only have two markers '*' for non-diff and '#' for frozen
    if pytc.is_nondiff_field(field_item) or pytc.is_treeclass_nondiff(node_item):
        return "--x"
    elif pytc.is_frozen_field(field_item) or pytc.is_treeclass_frozen(node_item):
        return "-..-"
    return default


PyTree = Any


def _bold_text(text: str) -> str:
    # bold a text in ansci code
    return "<b>" + text + "</b>"


def _tree_mermaid(tree: PyTree):
    def node_id(input):
        """hash a node by its location in a tree"""
        return ctypes.c_size_t(hash(input)).value

    def recurse(tree, depth, prev_id):
        if not pytc.is_treeclass(tree):
            return

        nonlocal FMT

        cur_id = node_id((depth, -1, prev_id))

        for i, field_item in enumerate(pytc.fields(tree)):

            if not field_item.repr or (pytc.is_nondiff_field(field_item)):
                continue

            node_item = getattr(tree, field_item.name)
            count, size = _reduce_count_and_size(node_item)

            if count.real == 0:
                count = size = ""
            else:
                count = (
                    _format_count(count.real)
                    + " param"
                    + ("s" if count.real > 1 else "")
                )
                size = _format_size(size.real)
            cur_id = node_id((depth, i, prev_id))

            if isinstance(node_item, src.tree_base._treeBase):
                mark = _marker(field_item, node_item, default="--->")
                FMT += f"\n\tid{prev_id} {mark} "
                FMT += f'|"{(count)}<br>{(size)}"| '
                FMT += f'id{cur_id}("{_bold_text(field_item.name) }<br>{node_item.__class__.__name__}")'
                recurse(tree=node_item, depth=depth + 1, prev_id=cur_id)
            else:
                mark = _marker(field_item, node_item, default="----")
                FMT += f"\n\tid{prev_id} {mark} "
                FMT += f'|"{(count)}<br>{(size)}"| '
                FMT += f'id{cur_id}["{_bold_text(field_item.name)}<br>{_format_node_diagram(node_item)}"]'

        prev_id = cur_id

    cur_id = node_id((0, 0, -1, 0))
    FMT = f"flowchart LR\n\tid{cur_id}({_bold_text(tree.__class__.__name__)})"
    recurse(tree=tree, depth=1, prev_id=cur_id)
    return FMT.expandtabs(4)


def tree_mermaid(tree, link=False):
    mermaid_string = _tree_mermaid(tree)
    return _generate_mermaid_link(mermaid_string) if link else mermaid_string
