from __future__ import annotations

import ctypes
from typing import Any

import jax.numpy as jnp
import requests

import pytreeclass
from pytreeclass.src.decorator_util import dispatch
from pytreeclass.src.tree_util import (
    _reduce_count_and_size,
    is_treeclass,
    is_treeclass_leaf,
    sequential_tree_shape_eval,
)
from pytreeclass.src.tree_viz_util import (
    _format_count,
    _format_node_diagram,
    _format_node_repr,
    _format_node_str,
    _format_size,
    _layer_box,
    _table,
    _vbox,
)

PyTree = Any


def tree_summary_md(tree: PyTree, array: jnp.ndarray | None = None) -> str:

    if array is not None:
        shape = sequential_tree_shape_eval(tree, array)
        indim_shape, outdim_shape = shape[:-1], shape[1:]

    def _cell(text):
        return f"<td align = 'center'> {text} </td>"

    def _leaf_info(tree_leaf: PyTree | Any) -> tuple[str, complex, complex]:
        """return (name, count, size) of a treeclass leaf / Any object"""

        @dispatch(argnum=0)
        def _info(leaf):
            """Any non-treeclass object"""
            count, size = _reduce_count_and_size(leaf)
            return (count, size)

        @_info.register(pytreeclass.src.tree_base.treeBase)
        def _(leaf):
            """treeclass leaf"""
            dynamic, static = leaf.__tree_fields__
            all_fields = {**dynamic, **static}
            count, size = _reduce_count_and_size(all_fields)
            return (count, size)

        return _info(tree_leaf)

    @dispatch(argnum=0)
    def recurse(tree, path=(), frozen_state=None):
        ...

    @recurse.register(pytreeclass.src.tree_base.treeBase)
    def _(tree, path=(), frozen_state=None):
        assert is_treeclass(tree)

        nonlocal FMT, COUNT, SIZE

        for i, fi in enumerate(tree.__dataclass_fields__.values()):

            cur_node = tree.__dict__[fi.name]

            if is_treeclass(cur_node) and not is_treeclass_leaf(cur_node):
                # Non leaf treeclass node
                recurse(
                    cur_node, path + (cur_node.__class__.__name__,), cur_node.frozen
                )

            elif is_treeclass_leaf(cur_node) or not is_treeclass(cur_node):
                # Leaf node (treeclass or non-treeclass)
                count, size = _leaf_info(cur_node)
                frozen_str = "<br>(frozen)" if frozen_state else ""
                name_str = f"{fi.name}{frozen_str}"
                type_str = "/".join(path + (cur_node.__class__.__name__,))
                count_str = _format_count(count, True)
                size_str = _format_size(size, True)
                config_str = (
                    "<br>".join(
                        [
                            f"{k}={_format_node_repr(v,0)}"
                            for k, v in cur_node.__tree_fields__[0].items()
                        ]
                    )
                    if is_treeclass(cur_node)
                    else f"{fi.name}={_format_node_repr(cur_node,0)}"
                )

                shape_str = (
                    f"{_format_node_repr(indim_shape[i],0)}\n{_format_node_repr(outdim_shape[i],0)}"
                    if array is not None
                    else ""
                )

                COUNT[1 if frozen_state else 0] += count
                SIZE[1 if frozen_state else 0] += size

                FMT += (
                    "<tr>"
                    + _cell(name_str)
                    + _cell(type_str)
                    + _cell(count_str)
                    + _cell(size_str)
                    + _cell(config_str)
                    + _cell(shape_str)
                    + "</tr>"
                )

    FMT = (
        "<table>\n"
        "<tr>\n"
        "<td align = 'center'> Name </td>\n"
        "<td align = 'center'> Type </td>\n"
        "<td align = 'center'> Param #</td>\n"
        "<td align = 'center'> Size </td>\n"
        "<td align = 'center'> Config </td>\n"
        "<td align = 'center'> Input/Output </td>\n"
        "</tr>\n"
    )

    COUNT = [0, 0]
    SIZE = [0, 0]

    recurse(tree, path=(), frozen_state=tree.frozen)

    FMT += "</table>"

    SUMMARY = (
        "<table>"
        f"<tr><td>Total #</td><td>{_format_count(sum(COUNT))}</td></tr>"
        f"<tr><td>Dynamic #</td><td>{_format_count(COUNT[0])}</td></tr>"
        f"<tr><td>Static/Frozen #</td><td>{_format_count(COUNT[1])}</td></tr>"
        f"<tr><td>Total size</td><td>{_format_size(sum(SIZE))}</td></tr>"
        f"<tr><td>Dynamic size</td><td>{_format_size(SIZE[0])}</td></tr>"
        f"<tr><td>Static/Frozen size</td><td>{_format_size(SIZE[1])}</td></tr>"
        "</table>"
    )

    return FMT + "\n\n#### Summary\n" + SUMMARY


def tree_summary(tree: PyTree, array: jnp.ndarray | None = None) -> str:

    if array is not None:
        shape = sequential_tree_shape_eval(tree, array)
        indim_shape, outdim_shape = shape[:-1], shape[1:]

    def _leaf_info(tree_leaf: PyTree | Any) -> tuple[str, complex, complex]:
        """return (name, count, size) of a treeclass leaf / Any object"""

        @dispatch(argnum=0)
        def _info(leaf):
            """Any object"""
            count, size = _reduce_count_and_size(leaf)
            return (count, size)

        @_info.register(pytreeclass.src.tree_base.treeBase)
        def _(leaf):
            """treeclass leaf"""
            dynamic, static = leaf.__tree_fields__
            all_fields = {**dynamic, **static}
            count, size = _reduce_count_and_size(all_fields)
            return (count, size)

        return _info(tree_leaf)

    @dispatch(argnum=0)
    def recurse(tree, path=(), frozen_state=None):
        ...

    @recurse.register(pytreeclass.src.tree_base.treeBase)
    def _(tree, path=(), frozen_state=None):
        assert is_treeclass(tree)
        _format_node = lambda node: _format_node_repr(node, depth=0).expandtabs(1)

        nonlocal ROWS, COUNT, SIZE

        for i, fi in enumerate(tree.__dataclass_fields__.values()):

            cur_node = tree.__dict__[fi.name]

            if is_treeclass(cur_node) and not is_treeclass_leaf(cur_node):
                # Non leaf treeclass node
                recurse(
                    cur_node, path + (cur_node.__class__.__name__,), cur_node.frozen
                )

            elif (
                is_treeclass_leaf(cur_node) or not is_treeclass(cur_node)
            ) and fi.repr:
                # Leaf node (treeclass or non-treeclass)
                count, size = _leaf_info(cur_node)
                frozen_str = "\n(frozen)" if frozen_state else ""
                name_str = f"{fi.name}{frozen_str}"
                type_str = "/".join(path + (cur_node.__class__.__name__,))
                count_str = _format_count(count, True)
                size_str = _format_size(size, True)
                config_str = (
                    "\n".join(
                        [
                            f"{k}={_format_node(v)}"
                            for k, v in cur_node.__tree_fields__[0].items()
                        ]
                    )
                    if is_treeclass(cur_node)
                    else f"{fi.name}={_format_node(cur_node)}"
                )

                shape_str = (
                    f"{_format_node(indim_shape[i])}\n{_format_node(outdim_shape[i])}"
                    if array is not None
                    else ""
                )

                COUNT[1 if frozen_state else 0] += count
                SIZE[1 if frozen_state else 0] += size

                ROWS.append(
                    [name_str, type_str, count_str, size_str, config_str, shape_str]
                )

    ROWS = [["Name", "Type ", "Param #", "Size ", "Config", "Input/Output"]]
    COUNT = [0, 0]
    SIZE = [0, 0]

    recurse(tree, path=(), frozen_state=tree.frozen)

    COLS = [list(c) for c in zip(*ROWS)]
    if array is None:
        COLS.pop()

    layer_table = _table(COLS)
    table_width = len(layer_table.split("\n")[0])

    param_summary = (
        f"Total # :\t\t{_format_count(sum(COUNT))}\n"
        f"Dynamic #:\t\t{_format_count(COUNT[0])}\n"
        f"Static/Frozen #:\t{_format_count(COUNT[1])}\n"
        f"{'-'*max([table_width,40])}\n"
        f"Total size :\t\t{_format_size(sum(SIZE))}\n"
        f"Dynamic size:\t\t{_format_size(SIZE[0])}\n"
        f"Static/Frozen size:\t{_format_size(SIZE[1])}\n"
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

            for fi in tree.__dataclass_fields__.values():
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

    @dispatch(argnum=1)
    def recurse_field(
        field_item, node_item, frozen_state, parent_level_count, node_index
    ):
        nonlocal FMT

        if field_item.repr:
            is_static_field = field_item.metadata.get("static", False)
            mark = "*" if is_static_field else ("#" if frozen_state else "─")
            is_last_field = node_index == 1

            FMT += "\n"
            FMT += "".join(
                [(("│" if lvl > 1 else "") + "\t") for lvl in parent_level_count]
            )

            FMT += f"└{mark}─ " if is_last_field else f"├{mark}─ "
            FMT += f"{field_item.name}"
            FMT += f"={_format_node_diagram(node_item)}"

        recurse(node_item, parent_level_count + [1], frozen_state)

    @recurse_field.register(pytreeclass.src.tree_base.treeBase)
    def _(field_item, node_item, frozen_state, parent_level_count, node_index):
        nonlocal FMT
        assert is_treeclass(node_item)

        if field_item.repr:
            frozen_state = node_item.frozen
            is_static_field = field_item.metadata.get("static", False)
            mark = "*" if is_static_field else ("#" if frozen_state else "─")
            layer_class_name = node_item.__class__.__name__

            is_last_field = node_index == 1

            FMT += "\n" + "".join(
                [(("│" if lvl > 1 else "") + "\t") for lvl in parent_level_count]
            )

            FMT += f"└{mark}─ " if is_last_field else f"├{mark}─ "
            FMT += f"{field_item.name}"
            FMT += f"={layer_class_name}"

            recurse(node_item, parent_level_count + [node_index], frozen_state)

    @dispatch(argnum=0)
    def recurse(tree, parent_level_count, frozen_state):
        ...

    @recurse.register(pytreeclass.src.tree_base.treeBase)
    def _(tree, parent_level_count, frozen_state):
        nonlocal FMT

        assert is_treeclass(tree)

        leaves_count = len(tree.__dataclass_fields__)

        for i, fi in enumerate(tree.__dataclass_fields__.values()):

            cur_node = tree.__dict__[fi.name]

            recurse_field(
                fi, cur_node, frozen_state, parent_level_count, leaves_count - i
            )
        FMT += "\t"

    FMT = f"{(tree.__class__.__name__)}"

    recurse(tree, [1], tree.frozen)

    return FMT.expandtabs(4)


def tree_repr(tree, width: int = 40) -> str:
    """Prertty print `treeclass_leaves`

    Returns:
        str: indented tree leaves.
    """

    def format_width(string, width=width):
        """strip newline/tab characters if less than max width"""
        stripped_string = string.replace("\n", "").replace("\t", "")
        children_length = len(stripped_string)
        return string if children_length > width else stripped_string

    @dispatch(argnum=1)
    def recurse_field(field_item, node_item, depth, frozen_state, is_last_field):
        """format non-treeclass field"""
        nonlocal FMT

        if field_item.repr:
            is_static_field = field_item.metadata.get("static", False)
            mark = "*" if is_static_field else ("#" if frozen_state else "")

            FMT += "\n" + "\t" * depth
            FMT += f"{mark}{field_item.name}"
            FMT += f"={format_width(_format_node_repr(node_item,depth))}"
            FMT += "" if is_last_field else ","

        recurse(node_item, depth, frozen_state)

    @recurse_field.register(pytreeclass.src.tree_base.treeBase)
    def _(field_item, node_item, depth, frozen_state, is_last_field):
        """format treeclass field"""
        nonlocal FMT
        assert is_treeclass(node_item)
        if field_item.repr:
            is_static_field = field_item.metadata.get("static", False)
            mark = "*" if is_static_field else ("#" if frozen_state else "")

            FMT += "\n" + "\t" * depth
            layer_class_name = f"{node_item.__class__.__name__}"

            FMT += f"{mark}{field_item.name}"
            FMT += f"={layer_class_name}" + "("

            start_cursor = len(FMT)  # capture children repr

            recurse(node_item, depth=depth + 1, frozen_state=node_item.frozen)

            FMT = FMT[:start_cursor] + format_width(FMT[start_cursor:]) + ")"
            FMT += "" if is_last_field else ","

    @dispatch(argnum=0)
    def recurse(tree, depth, frozen_state):
        ...

    @recurse.register(pytreeclass.src.tree_base.treeBase)
    def _(tree, depth, frozen_state):
        nonlocal FMT
        is_treeclass(tree)

        leaves_count = len(tree.__dataclass_fields__)
        for i, fi in enumerate(tree.__dataclass_fields__.values()):

            # retrieve node item
            cur_node = tree.__dict__[fi.name]

            recurse_field(
                fi,
                cur_node,
                depth,
                frozen_state,
                True if i == (leaves_count - 1) else False,
            )

    FMT = ""
    recurse(tree, depth=1, frozen_state=tree.frozen)
    FMT = f"{(tree.__class__.__name__)}({format_width(FMT,width)})"

    return FMT.expandtabs(2)


def tree_str(tree, width: int = 40) -> str:
    """Prertty print `treeclass_leaves`

    Returns:
        str: indented tree leaves.
    """

    def format_width(string, width=width):
        """strip newline/tab characters if less than max width"""
        stripped_string = string.replace("\n", "").replace("\t", "")
        children_length = len(stripped_string)
        return string if children_length > width else stripped_string

    @dispatch(argnum=1)
    def recurse_field(field_item, node_item, depth, frozen_state, is_last_field):
        """format non-treeclass field"""
        nonlocal FMT

        if field_item.repr:
            is_static_field = field_item.metadata.get("static", False)
            mark = "*" if is_static_field else ("#" if frozen_state else "")

            FMT += "\n" + "\t" * depth
            FMT += f"{mark}{field_item.name}"
            FMT += f"={format_width(_format_node_str(node_item,depth))}"
            FMT += "" if is_last_field else ","

        recurse(node_item, depth, frozen_state)

    @recurse_field.register(pytreeclass.src.tree_base.treeBase)
    def _(field_item, node_item, depth, frozen_state, is_last_field):
        """format treeclass field"""
        nonlocal FMT
        assert is_treeclass(node_item)

        if field_item.repr:
            is_static_field = field_item.metadata.get("static", False)
            mark = "*" if is_static_field else ("#" if frozen_state else "")

            FMT += "\n" + "\t" * depth
            layer_class_name = f"{node_item.__class__.__name__}"

            FMT += f"{mark}{field_item.name}"
            FMT += f"={layer_class_name}" + "("
            start_cursor = len(FMT)  # capture children repr

            recurse(node_item, depth=depth + 1, frozen_state=node_item.frozen)

            FMT = FMT[:start_cursor] + format_width(FMT[start_cursor:]) + ")"
            FMT += "" if is_last_field else ","

    @dispatch(argnum=0)
    def recurse(tree, depth, frozen_state):
        ...

    @recurse.register(pytreeclass.src.tree_base.treeBase)
    def _(tree, depth, frozen_state):
        nonlocal FMT
        assert is_treeclass(tree)

        leaves_count = len(tree.__dataclass_fields__)
        for i, fi in enumerate(tree.__dataclass_fields__.values()):

            # retrieve node item
            cur_node = tree.__dict__[fi.name]

            recurse_field(
                fi,
                cur_node,
                depth,
                frozen_state,
                True if i == (leaves_count - 1) else False,
            )

    FMT = ""
    recurse(tree, depth=1, frozen_state=tree.frozen)
    FMT = f"{(tree.__class__.__name__)}({format_width(FMT,width)})"

    return FMT.expandtabs(2)


def _tree_mermaid(tree):
    def node_id(input):
        """hash a node by its location in a tree"""
        return ctypes.c_size_t(hash(input)).value

    @dispatch(argnum=1)
    def recurse_field(field_item, node_item, depth, prev_id, order, frozen_state):
        nonlocal FMT

        if field_item.repr:
            # create node id from depth, order, and previous id
            cur_id = node_id((depth, order, prev_id))
            mark = (
                "--x"
                if field_item.metadata.get("static", False)
                else ("-.-" if frozen_state else "---")
            )
            FMT += f'\n\tid{prev_id} {mark} id{cur_id}["{field_item.name}\\n{_format_node_diagram(node_item)}"]'
            prev_id = cur_id

        recurse(node_item, depth, prev_id, frozen_state)

    @recurse_field.register(pytreeclass.src.tree_base.treeBase)
    def _(field_item, node_item, depth, prev_id, order, frozen_state):
        nonlocal FMT
        assert is_treeclass(node_item)

        if field_item.repr:
            layer_class_name = node_item.__class__.__name__
            cur_id = node_id((depth, order, prev_id))
            FMT += f"\n\tid{prev_id} --> id{cur_id}({field_item.name}\\n{layer_class_name})"
            recurse(node_item, depth + 1, cur_id, node_item.frozen)

    @dispatch(argnum=0)
    def recurse(tree, depth, prev_id, frozen_state):
        ...

    @recurse.register(pytreeclass.src.tree_base.treeBase)
    def _(tree, depth, prev_id, frozen_state):
        nonlocal FMT
        assert is_treeclass(tree)

        for i, fi in enumerate(tree.__dataclass_fields__.values()):

            # retrieve node item
            cur_node = tree.__dict__[fi.name]

            recurse_field(
                fi,
                cur_node,
                depth,
                prev_id,
                i,
                frozen_state,
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

    elif method == "summary_md":
        with open(f"{filename}.md", "w") as f:
            f.write(tree_summary_md(tree))
