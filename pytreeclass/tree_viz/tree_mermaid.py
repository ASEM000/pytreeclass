from __future__ import annotations

import ctypes
from dataclasses import Field
from typing import Any

import pytreeclass._src as src
from pytreeclass._src.tree_util import (
    _tree_fields,
    is_frozen_field,
    is_nondiff_field,
    is_treeclass,
    is_treeclass_frozen,
    is_treeclass_nondiff,
)
from pytreeclass.tree_viz.node_pprint import _format_node_diagram
from pytreeclass.tree_viz.tree_export import _generate_mermaid_link


def _marker(field_item: Field, node_item: Any, default: str = "---") -> str:
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
        return "--x"
    elif is_frozen_field(field_item) or is_treeclass_frozen(node_item):
        return "-.-"
    else:
        return default


PyTree = Any


def _tree_mermaid(tree: PyTree):
    def node_id(input):
        """hash a node by its location in a tree"""
        return ctypes.c_size_t(hash(input)).value

    # @dispatch(argnum="node_item")
    def recurse_field(field_item, node_item, depth, prev_id, order):
        nonlocal FMT

        if not field_item.repr:
            return

        if isinstance(node_item, src.tree_base._treeBase):
            layer_class_name = node_item.__class__.__name__
            cur_id = node_id((depth, order, prev_id))
            mark = _marker(field_item, node_item, default="-->")
            FMT += f"\n\tid{prev_id} {mark} id{cur_id}({field_item.name}\\n{layer_class_name})"
            recurse(tree=node_item, depth=depth + 1, prev_id=cur_id)

        else:
            # create node id from depth, order, and previous id
            cur_id = node_id((depth, order, prev_id))
            mark = _marker(field_item, node_item)
            FMT += f'\n\tid{prev_id} {mark} id{cur_id}["{field_item.name}\\n{_format_node_diagram(node_item)}"]'
            prev_id = cur_id

        recurse(tree=node_item, depth=depth, prev_id=prev_id)

    def recurse(tree, depth, prev_id):
        if not is_treeclass(tree):
            return

        nonlocal FMT

        for i, fi in enumerate(_tree_fields(tree).values()):
            recurse_field(fi, getattr(tree, fi.name), depth, prev_id, i)

    cur_id = node_id((0, 0, -1, 0))
    FMT = f"flowchart LR\n\tid{cur_id}[{tree.__class__.__name__}]"
    recurse(tree=tree, depth=1, prev_id=cur_id)
    return FMT.expandtabs(4)


def tree_mermaid(tree, link=False):
    mermaid_string = _tree_mermaid(tree)
    return _generate_mermaid_link(mermaid_string) if link else mermaid_string
