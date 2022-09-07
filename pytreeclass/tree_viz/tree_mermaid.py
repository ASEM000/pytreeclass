from __future__ import annotations

import ctypes

import pytreeclass._src as src
from pytreeclass._src.dispatch import dispatch
from pytreeclass._src.tree_util import _tree_fields, is_treeclass_frozen
from pytreeclass.tree_viz.node_pprint import _format_node_diagram
from pytreeclass.tree_viz.tree_export import _generate_mermaid_link


def _tree_mermaid(tree):
    def node_id(input):
        """hash a node by its location in a tree"""
        return ctypes.c_size_t(hash(input)).value

    @dispatch(argnum=1)
    def recurse_field(field_item, node_item, depth, prev_id, order, is_frozen):
        nonlocal FMT

        if field_item.repr:
            # create node id from depth, order, and previous id
            cur_id = node_id((depth, order, prev_id))
            mark = (
                "--x"
                if field_item.metadata.get("static", False)
                else ("-.-" if is_frozen else "---")
            )
            FMT += f'\n\tid{prev_id} {mark} id{cur_id}["{field_item.name}\\n{_format_node_diagram(node_item)}"]'
            prev_id = cur_id

        recurse(node_item, depth, prev_id, is_frozen)

    @recurse_field.register(src.tree_base._treeBase)
    def _(field_item, node_item, depth, prev_id, order, is_frozen):
        nonlocal FMT

        if field_item.repr:
            layer_class_name = node_item.__class__.__name__
            cur_id = node_id((depth, order, prev_id))
            FMT += f"\n\tid{prev_id} --> id{cur_id}({field_item.name}\\n{layer_class_name})"
            recurse(node_item, depth + 1, cur_id, is_treeclass_frozen(node_item))

    @dispatch(argnum=0)
    def recurse(tree, depth, prev_id, is_frozen):
        ...

    @recurse.register(src.tree_base._treeBase)
    def _(tree, depth, prev_id, is_frozen):
        nonlocal FMT

        for i, fi in enumerate(_tree_fields(tree).values()):

            # retrieve node item
            cur_node = tree.__dict__[fi.name]

            recurse_field(
                fi,
                cur_node,
                depth,
                prev_id,
                i,
                is_frozen,
            )

    cur_id = node_id((0, 0, -1, 0))
    FMT = f"flowchart LR\n\tid{cur_id}[{tree.__class__.__name__}]"
    recurse(tree, 1, cur_id, is_treeclass_frozen(tree))
    return FMT.expandtabs(4)


def tree_mermaid(tree, link=False):
    mermaid_string = _tree_mermaid(tree)
    return _generate_mermaid_link(mermaid_string) if link else mermaid_string
