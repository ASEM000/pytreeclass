from __future__ import annotations

import ctypes
import os
from dataclasses import field
from typing import Any

import jax.numpy as jnp
import requests

import pytreeclass.src as src
from pytreeclass.src.decorator_util import dispatch

# from pytreeclass.src.misc import static_value
from pytreeclass.src.tree_util import (
    _reduce_count_and_size,
    is_treeclass,
    is_treeclass_leaf,
    is_treeclass_non_leaf,
    sequential_tree_shape_eval,
)
from pytreeclass.src.tree_viz_util import (
    _format_count,
    _format_node_diagram,
    _format_node_repr,
    _format_node_str,
    _format_size,
    _format_width,
    _layer_box,
    _table,
    _vbox,
)

PyTree = Any


# def tree_summary_md(tree: PyTree, array: jnp.ndarray | None = None) -> str:

#     if array is not None:
#         shape = sequential_tree_shape_eval(tree, array)
#         indim_shape, outdim_shape = shape[:-1], shape[1:]

#     def _cell(text):
#         return f"<td align = 'center'> {text} </td>"

#     @dispatch(argnum=0)
#     def recurse(tree, path=(), is_frozen=None):
#         ...

#     @recurse.register(src.tree_base._treeBase)
#     def _(tree, path=(), is_frozen=None):
#         assert is_treeclass(tree)

#         nonlocal FMT, COUNT, SIZE
#         all_fields = {
#             **tree.__dataclass_fields__,
#             **tree.__dict__.get("__pytree_fields__", {}),
#         }

#         for i, fi in enumerate(all_fields.values()):

#             cur_node = tree.__dict__[fi.name]

#             if is_treeclass_non_leaf(cur_node):
#                 class_name = cur_node.__class__.__name__
#                 recurse(cur_node, path + (class_name,), cur_node.frozen)

#             elif fi.repr:
#                 # Leaf node (treeclass or non-treeclass)
#                 count, size = _leaf_info(cur_node)
#                 frozen_str = "<br>(frozen)" if is_frozen else ""
#                 name_str = f"{fi.name}{frozen_str}"
#                 type_str = "/".join(path + (cur_node.__class__.__name__,))
#                 count_str = _format_count(count, True)
#                 size_str = _format_size(size, True)
#                 config_str = (
#                     "<br>".join(
#                         [
#                             f"{k}={_format_node_repr(v,0)}"
#                             for k, v in cur_node.__tree_fields__[0].items()
#                         ]
#                     )
#                     if is_treeclass(cur_node)
#                     else f"{fi.name}={_format_node_repr(cur_node,0)}"
#                 )

#                 shape_str = (
#                     f"{_format_node_repr(indim_shape[i],0)}\n{_format_node_repr(outdim_shape[i],0)}"
#                     if array is not None
#                     else ""
#                 )

#                 COUNT[1 if is_frozen else 0] += count
#                 SIZE[1 if is_frozen else 0] += size

#                 FMT += (
#                     "<tr>"
#                     + _cell(name_str)
#                     + _cell(type_str)
#                     + _cell(count_str)
#                     + _cell(size_str)
#                     + _cell(config_str)
#                     + _cell(shape_str)
#                     + "</tr>"
#                 )

#     FMT = (
#         "<table>\n"
#         "<tr>\n"
#         "<td align = 'center'> Name </td>\n"
#         "<td align = 'center'> Type </td>\n"
#         "<td align = 'center'> Param #</td>\n"
#         "<td align = 'center'> Size </td>\n"
#         "<td align = 'center'> Config </td>\n"
#         "<td align = 'center'> Input/Output </td>\n"
#         "</tr>\n"
#     )

#     COUNT = [0, 0]
#     SIZE = [0, 0]

#     recurse(tree, path=(), is_frozen=tree.frozen)

#     FMT += "</table>"

#     SUMMARY = (
#         "<table>"
#         f"<tr><td>Total #</td><td>{_format_count(sum(COUNT))}</td></tr>"
#         f"<tr><td>Dynamic #</td><td>{_format_count(COUNT[0])}</td></tr>"
#         f"<tr><td>Static/Frozen #</td><td>{_format_count(COUNT[1])}</td></tr>"
#         f"<tr><td>Total size</td><td>{_format_size(sum(SIZE))}</td></tr>"
#         f"<tr><td>Dynamic size</td><td>{_format_size(SIZE[0])}</td></tr>"
#         f"<tr><td>Static/Frozen size</td><td>{_format_size(SIZE[1])}</td></tr>"
#         "</table>"
#     )

#     return FMT + "\n\n#### Summary\n" + SUMMARY


def tree_summary(tree, array: jnp.ndarray = None) -> str:
    _format_node = lambda node: _format_node_repr(node, depth=0).expandtabs(1)

    if array is not None:
        shape = sequential_tree_shape_eval(tree, array)
        indim_shape, outdim_shape = shape[:-1], shape[1:]

        shape_str = ["Input/Output"] + [
            f"{_format_node(indim_shape[i])}\n{_format_node(outdim_shape[i])}"
            for i in range(len(indim_shape))
        ]

    @dispatch(argnum="node_item")
    def recurse_field(field_item, node_item, is_frozen, name_path, type_path):
        ...

    @recurse_field.register(int)
    @recurse_field.register(float)
    @recurse_field.register(complex)
    @recurse_field.register(str)
    @recurse_field.register(bool)
    @recurse_field.register(jnp.ndarray)
    def _(field_item, node_item, is_frozen, name_path, type_path):
        nonlocal ROWS, COUNT, SIZE

        if field_item.repr:
            count, size = _reduce_count_and_size(node_item)
            ROWS.append(
                [
                    "/".join(name_path) + f"{('(frozen)' if is_frozen else '')}",
                    "/".join(type_path),
                    _format_count(count),
                    _format_size(size, True),
                    f"{field_item.name}={_format_node(node_item)}",
                ]
            )

            # non-treeclass leaf inherit frozen state
            COUNT[1 if is_frozen else 0] += count
            SIZE[1 if is_frozen else 0] += size

    @recurse_field.register(list)
    @recurse_field.register(tuple)
    def _(field_item, node_item, is_frozen, name_path, type_path):

        if field_item.repr:

            for i, layer in enumerate(node_item):
                new_field = field()
                object.__setattr__(new_field, "name", f"{field_item.name}_{i}")
                object.__setattr__(new_field, "type", type(layer))

                recurse_field(
                    field_item=new_field,
                    node_item=layer,
                    is_frozen=is_frozen,
                    name_path=name_path + (f"{field_item.name}_{i}",),
                    type_path=type_path + (layer.__class__.__name__,),
                )

    @recurse_field.register(src.tree_base._treeBase)
    def _(field_item, node_item, is_frozen, name_path, type_path):
        nonlocal ROWS, COUNT, SIZE

        if field_item.repr:
            is_frozen = node_item.frozen
            count, size = _reduce_count_and_size(node_item.at[...].unfreeze())
            dynamic, _ = node_item.__pytree_structure__
            ROWS.append(
                [
                    "/".join(name_path)
                    + f"{(os.linesep + '(frozen)' if is_frozen else '')}",
                    "/".join(type_path),
                    _format_count(count),
                    _format_size(size, True),
                    "\n".join([f"{k}={_format_node(v)}" for k, v in dynamic.items()]),
                ]
            )

            COUNT[1 if is_frozen else 0] += count
            SIZE[1 if is_frozen else 0] += size

    def recurse(tree, is_frozen, name_path, type_path):

        nonlocal ROWS, COUNT, SIZE

        for field_item in tree.__pytree_fields__.values():
            node_item = tree.__dict__[field_item.name]

            if is_treeclass_non_leaf(node_item):
                recurse(
                    tree=node_item,
                    is_frozen=node_item.frozen,
                    name_path=name_path + (field_item.name,),
                    type_path=type_path + (node_item.__class__.__name__,),
                )

            else:

                is_static = field_item.metadata.get("static", False)

                if not (is_static):
                    recurse_field(
                        field_item=field_item,
                        node_item=node_item,
                        is_frozen=is_frozen,
                        name_path=name_path + (field_item.name,),
                        type_path=type_path + (node_item.__class__.__name__,),
                    )

    ROWS = [["Name", "Type ", "Param #", "Size ", "Config"]]
    COUNT = [0, 0]
    SIZE = [0, 0]

    recurse(tree, is_frozen=tree.frozen, name_path=(), type_path=())

    COLS = [list(c) for c in zip(*ROWS)]

    if array is not None:
        COLS += [shape_str]

    layer_table = _table(COLS)
    table_width = len(layer_table.split("\n")[0])

    param_summary = (
        f"Total count :\t{_format_count(sum(COUNT))}\n"
        f"Dynamic count :\t{_format_count(COUNT[0])}\n"
        f"Frozen count :\t{_format_count(COUNT[1])}\n"
        f"{'-'*max([table_width,40])}\n"
        f"Total size :\t{_format_size(sum(SIZE))}\n"
        f"Dynamic size :\t{_format_size(SIZE[0])}\n"
        f"Frozen size :\t{_format_size(SIZE[1])}\n"
        f"{'='*max([table_width,40])}"
    )

    return layer_table + "\n" + param_summary


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


def tree_diagram(tree):
    """
    === Explanation
        pretty print treeclass tree with tree structure diagram

    === Args
        tree : boolean to create tree-structure
    """

    @dispatch(argnum="node_item")
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
                field_item=field_item,
                node_item=node_item.__class__,
                is_frozen=is_frozen,
                parent_level_count=parent_level_count,
                node_index=node_index,
            )

            for i, layer in enumerate(node_item):
                new_field = field()
                object.__setattr__(new_field, "name", f"{field_item.name}_{i}")
                object.__setattr__(new_field, "type", type(layer))

                recurse_field(
                    field_item=new_field,
                    node_item=layer,
                    is_frozen=is_frozen,
                    parent_level_count=parent_level_count + [node_index],
                    node_index=len(node_item) - i,
                )

        recurse(node_item, parent_level_count, is_frozen)

    @recurse_field.register(src.tree_base._treeBase)
    def _(field_item, node_item, is_frozen, parent_level_count, node_index):
        nonlocal FMT

        if field_item.repr:
            is_frozen = node_item.frozen
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

    @dispatch(argnum=0)
    def recurse(tree, parent_level_count, is_frozen):
        ...

    @recurse.register(src.tree_base._treeBase)
    def _(tree, parent_level_count, is_frozen):
        nonlocal FMT

        leaves_count = len(tree.__pytree_fields__)

        for i, fi in enumerate(tree.__pytree_fields__.values()):
            cur_node = tree.__dict__[fi.name]

            recurse_field(
                field_item=fi,
                node_item=cur_node,
                is_frozen=is_frozen,
                parent_level_count=parent_level_count,
                node_index=leaves_count - i,
            )

        FMT += "\t"

    FMT = f"{(tree.__class__.__name__)}"

    recurse(tree, [1], tree.frozen)

    return FMT.expandtabs(4)


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
            is_frozen = node_item.frozen
            is_static = field_item.metadata.get("static", False)
            mark = "*" if is_static else ("#" if is_frozen else "")

            FMT += "\n" + "\t" * depth
            layer_class_name = f"{node_item.__class__.__name__}"

            FMT += f"{mark}{field_item.name}"
            FMT += f"={layer_class_name}" + "("
            start_cursor = len(FMT)  # capture children repr

            recurse(node_item, depth=depth + 1, is_frozen=node_item.frozen)

            FMT = FMT[:start_cursor] + _format_width(
                FMT[start_cursor:] + "\n" + "\t" * (depth) + ")"
            )
            FMT += "" if is_last_field else ","

    @dispatch(argnum=0)
    def recurse(tree, depth, is_frozen):
        ...

    @recurse.register(src.tree_base._treeBase)
    def _(tree, depth, is_frozen):
        nonlocal FMT

        leaves_count = len(tree.__pytree_fields__)
        for i, fi in enumerate(tree.__pytree_fields__.values()):

            # retrieve node item
            cur_node = getattr(tree, fi.name)

            recurse_field(
                fi,
                cur_node,
                depth,
                is_frozen,
                True if i == (leaves_count - 1) else False,
            )

    FMT = ""
    recurse(tree, depth=1, is_frozen=tree.frozen)
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
            is_frozen = node_item.frozen
            is_static = field_item.metadata.get("static", False)
            mark = "*" if is_static else ("#" if is_frozen else "")

            FMT += "\n" + "\t" * depth
            layer_class_name = f"{node_item.__class__.__name__}"

            FMT += f"{mark}{field_item.name}"
            FMT += f"={layer_class_name}" + "("
            start_cursor = len(FMT)  # capture children repr

            recurse(node_item, depth=depth + 1, is_frozen=node_item.frozen)

            FMT = FMT[:start_cursor] + _format_width(
                FMT[start_cursor:] + "\n" + "\t" * (depth) + ")"
            )
            FMT += "" if is_last_field else ","

    @dispatch(argnum=0)
    def recurse(tree, depth, is_frozen):
        ...

    @recurse.register(src.tree_base._treeBase)
    def _(tree, depth, is_frozen):
        nonlocal FMT

        leaves_count = len(tree.__pytree_fields__)
        for i, fi in enumerate(tree.__pytree_fields__.values()):

            # retrieve node item
            cur_node = tree.__dict__[fi.name]

            recurse_field(
                fi,
                cur_node,
                depth,
                is_frozen,
                True if i == (leaves_count - 1) else False,
            )

    FMT = ""
    recurse(tree, depth=1, is_frozen=tree.frozen)
    FMT = f"{(tree.__class__.__name__)}(" + _format_width(FMT + "\n)", width)

    return FMT.expandtabs(2)


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
            recurse(node_item, depth + 1, cur_id, node_item.frozen)

    @dispatch(argnum=0)
    def recurse(tree, depth, prev_id, is_frozen):
        ...

    @recurse.register(src.tree_base._treeBase)
    def _(tree, depth, prev_id, is_frozen):
        nonlocal FMT

        for i, fi in enumerate(tree.__pytree_fields__.values()):

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
    recurse(tree, 1, cur_id, tree.frozen)
    return FMT.expandtabs(4)


def _generate_mermaid_link(mermaid_string: str) -> str:
    """generate a one-time link mermaid diagram"""
    url_val = "https://pytreeclass.herokuapp.com/generateTemp"
    request = requests.post(url_val, json={"description": mermaid_string})
    generated_id = request.json()["id"]
    generated_html = f"https://pytreeclass.herokuapp.com/temp/?id={generated_id}"
    return f"Open URL in browser: {generated_html}"


def tree_mermaid(tree, link=False):
    mermaid_string = _tree_mermaid(tree)
    return _generate_mermaid_link(mermaid_string) if link else mermaid_string


def save_viz(tree, filename, method="tree_mermaid_md"):

    if method == "tree_mermaid_md":
        FMT = "```mermaid\n" + tree_mermaid(tree) + "\n```"

        with open(f"{filename}.md", "w") as f:
            f.write(FMT)

    elif method == "tree_mermaid_html":
        FMT = "<html><body><script src='https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js'></script>"
        FMT += "<script>mermaid.initialize({ startOnLoad: true });</script><div class='mermaid'>"
        FMT += tree_mermaid(tree)
        FMT += "</div></body></html>"

        with open(f"{filename}.html", "w") as f:
            f.write(FMT)

    elif method == "tree_diagram":
        with open(f"{filename}.txt", "w") as f:
            f.write(tree_diagram(tree))

    elif method == "tree_box":
        with open(f"{filename}.txt", "w") as f:
            f.write(tree_box(tree))

    elif method == "summary":
        with open(f"{filename}.txt", "w") as f:
            f.write(tree_summary(tree))

    # elif method == "summary_md":
    #     with open(f"{filename}.md", "w") as f:
    #         f.write(tree_summary_md(tree))
