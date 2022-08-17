from __future__ import annotations

import ctypes
import math
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import requests

import pytreeclass
from pytreeclass.src.decorator_util import dispatch
from pytreeclass.src.tree_util import (
    _reduce_count_and_size,
    is_treeclass,
    is_treeclass_leaf,
    sequential_tree_shape_eval,
)

PyTree = Any


# Node formatting


def _format_size(node_size, newline=False):
    """return formatted size from inexact(exact) complex number"""
    mark = "\n" if newline else ""
    order_kw = ["B", "KB", "MB", "GB"]

    # define order of magnitude
    real_size_order = int(math.log(node_size.real, 1024)) if node_size.real > 0 else 0
    imag_size_order = int(math.log(node_size.imag, 1024)) if node_size.imag > 0 else 0
    return (
        f"{(node_size.real)/(1024**real_size_order):.2f}{order_kw[real_size_order]}{mark}"
        f"({(node_size.imag)/(1024**imag_size_order):.2f}{order_kw[imag_size_order]})"
    )


def _format_count(node_count, newline=False):
    mark = "\n" if newline else ""
    return f"{int(node_count.real):,}{mark}({int(node_count.imag):,})"


def _format_node_repr(node, *args, **kwargs):
    @dispatch(argnum=0)
    def __format_node_repr(node, *args, **kwargs):
        return f"{node!r}"

    @__format_node_repr.register(jnp.ndarray)
    @__format_node_repr.register(jax.ShapeDtypeStruct)
    def _(node, *args, **kwargs):
        replace_tuple = (
            ("int", "i"),
            ("float", "f"),
            ("complex", "c"),
            (",)", ")"),
            ("(", "["),
            (")", "]"),
            (" ", ""),
        )

        formatted_string = f"{node.dtype}{jnp.shape(node)!r}"

        for lhs, rhs in replace_tuple:
            formatted_string = formatted_string.replace(lhs, rhs)
        return formatted_string

    @__format_node_repr.register(list)
    def _(node, depth=0):
        return (
            "[\n"
            + ",\n".join(["\t" * (depth + 1) + __format_node_repr(k) for k in node])
            + "]"
        )

    @__format_node_repr.register(tuple)
    def _(node, depth=0):
        return (
            "(\n"
            + ",\n".join(["\t" * (depth + 1) + __format_node_repr(k) for k in node])
            + ")"
        )

    @__format_node_repr.register(dict)
    def _(node, depth=0):
        return (
            "{\n"
            + ",\n".join(
                [
                    "\t" * (depth + 1) + f"{k}:{__format_node_repr(v)}"
                    for k, v in node.items()
                ]
            )
            + "}"
        )

    return __format_node_repr(node, *args, **kwargs)


def _format_node_str(node, *args, **kwargs):
    @dispatch(argnum=0)
    def __format_node_str(node, depth=0):
        multiline = "\n" in f"{node!s}"
        string = ("\n" + "\t" * (depth + 3)) if multiline else ""
        string += ("\n" + "\t" * (depth + 3)).join(f"{node!s}".split("\n"))
        return string

    @__format_node_str.register(list)
    def _(node, depth=0):
        string = ",\n".join(f"{layer!s}" for layer in node)
        shifted = "\t" * (depth + 3) + ("\n" + "\t" * (depth + 3)).join(
            string.split("\n")
        )
        return "[\n" + shifted + "]"

    @__format_node_str.register(tuple)
    def _(node, depth=0):
        string = ",\n".join(f"{layer!s}" for layer in node)
        shifted = "\t" * (depth + 3) + ("\n" + "\t" * (depth + 3)).join(
            string.split("\n")
        )
        return "(\n" + shifted + ")"

    @__format_node_str.register(dict)
    def _(node, depth=0):
        string = ",\n".join(f"{k}:{v!s}" for k, v in node.items())
        shifted = "\t" * (depth + 3) + ("\n" + "\t" * (depth + 3)).join(
            string.split("\n")
        )
        return "{\n" + shifted + "}"

    return __format_node_str(node, depth=0)


def _format_node_diagram(node, *args, **kwargs):
    @dispatch(argnum=0)
    def __format_node_diagram(node, *args, **kwargs):
        return f"{node!r}"

    @__format_node_diagram.register(jnp.ndarray)
    @__format_node_diagram.register(jax.ShapeDtypeStruct)
    def _(node, *args, **kwargs):
        replace_tuple = (
            ("int", "i"),
            ("float", "f"),
            ("complex", "c"),
            (",)", ")"),
            ("(", "["),
            (")", "]"),
            (" ", ""),
        )

        formatted_string = f"{node.dtype}{jnp.shape(node)!r}"

        for lhs, rhs in replace_tuple:
            formatted_string = formatted_string.replace(lhs, rhs)
        return formatted_string

    return __format_node_diagram(node, *args, **kwargs)


# Box drawing


def _hbox(*text):

    boxes = list(map(_vbox, text))
    boxes = [(box).split("\n") for box in boxes]
    max_col_height = max([len(b) for b in boxes])
    boxes = [b + [" " * len(b[0])] * (max_col_height - len(b)) for b in boxes]
    FMT = ""

    for _, line in enumerate(zip(*boxes)):
        FMT += _resolve_line(line) + "\n"
    return FMT


def _hstack(boxes):

    boxes = [(box).split("\n") for box in boxes]
    max_col_height = max([len(b) for b in boxes])

    # expand height of each col before merging
    boxes = [b + [" " * len(b[0])] * (max_col_height - len(b)) for b in boxes]

    FMT = ""

    _cells = tuple(zip(*boxes))

    for i, line in enumerate(_cells):
        FMT += _resolve_line(line) + ("\n" if i != (len(_cells) - 1) else "")

    return FMT


def _vbox(*text: tuple[str, ...]) -> str:
    """Create vertically stacked text boxes

    Returns:
        str: stacked boxes string

    Examples:
    >>> _vbox("a","b")
        ┌───┐
        │a  │
        ├───┤
        │b  │
        └───┘

    >>> _vbox("a","","a")
        ┌───┐
        │a  │
        ├───┤
        │   │
        ├───┤
        │a  │
        └───┘
    """

    max_width = (
        max(jtu.tree_flatten([[len(t) for t in item.split("\n")] for item in text])[0])
        + 0
    )

    top = f"┌{'─'*max_width}┐"
    line = f"├{'─'*max_width}┤"
    side = [
        "\n".join([f"│{t}{' '*(max_width-len(t))}│" for t in item.split("\n")])
        for item in text
    ]
    btm = f"└{'─'*max_width}┘"

    formatted = ""

    for i, s in enumerate(side):

        if i == 0:
            formatted += f"{top}\n{s}\n{line if len(side)>1 else btm}"

        elif i == len(side) - 1:
            formatted += f"\n{s}\n{btm}"

        else:
            formatted += f"\n{s}\n{line}"

    return formatted


def _resolve_line(cols: Sequence[str, ...]) -> str:
    """combine columns of single line by merging their borders

    Args:
        cols (Sequence[str,...]): Sequence of single line column string

    Returns:
        str: resolved column string

    Example:
        >>> _resolve_line(['ab','b│','│c'])
        'abb│c'

        >>> _resolve_line(['ab','b┐','┌c'])
        'abb┬c'

    """

    cols = list(map(list, cols))  # convert each col to col of chars
    alpha = ["│", "┌", "┐", "└", "┘", "┤", "├"]

    for index in range(len(cols) - 1):

        if cols[index][-1] == "┐" and cols[index + 1][0] in ["┌", "─"]:
            cols[index][-1] = "┬"
            cols[index + 1].pop(0)

        elif cols[index][-1] == "┘" and cols[index + 1][0] in ["└", "─"]:
            cols[index][-1] = "┴"
            cols[index + 1].pop(0)

        elif cols[index][-1] == "┤" and cols[index + 1][0] in ["├", "─", "└"]:  #
            cols[index][-1] = "┼"
            cols[index + 1].pop(0)

        elif cols[index][-1] in ["┘", "┐", "─"] and cols[index + 1][0] in ["├"]:
            cols[index][-1] = "┼"
            cols[index + 1].pop(0)

        elif cols[index][-1] == "─" and cols[index + 1][0] == "└":
            cols[index][-1] = "┴"
            cols[index + 1].pop(0)

        elif cols[index][-1] == "─" and cols[index + 1][0] == "┌":
            cols[index][-1] = "┬"
            cols[index + 1].pop(0)

        elif cols[index][-1] == "│" and cols[index + 1][0] == "─":
            cols[index][-1] = "├"
            cols[index + 1].pop(0)

        elif cols[index][-1] == " ":
            cols[index].pop()

        elif cols[index][-1] in alpha and cols[index + 1][0] in [*alpha, " "]:
            cols[index + 1].pop(0)

    return "".join(map(lambda x: "".join(x), cols))


def _table(lines):
    """

    === Explanation
        create a _table with self aligning rows and cols

    === Args
        lines : list of lists of cols values

    === Examples
        >>> print(_table([['1\n','2'],['3','4000']]))
            ┌─┬────────┐
            │1│3       │
            │ │        │
            ├─┼────────┤
            │2│40000000│
            └─┴────────┘


    """
    # align _cells vertically
    for i, _cells in enumerate(zip(*lines)):
        max__cell_height = max(map(lambda x: x.count("\n"), _cells))
        for j in range(len(_cells)):
            lines[j][i] += "\n" * (max__cell_height - lines[j][i].count("\n"))
    cols = [_vbox(*col) for col in lines]

    return _hstack(cols)


def _layer_box(name, indim=None, outdim=None):
    """
    === Explanation
        create a keras-like layer diagram

    ==== Examples
        >>> print(_layer_box("Test",(1,1,1),(1,1,1)))
        ┌──────┬────────┬───────────┐
        │      │ Input  │ (1, 1, 1) │
        │ Test │────────┼───────────┤
        │      │ Output │ (1, 1, 1) │
        └──────┴────────┴───────────┘

    """

    return _hstack(
        [
            _vbox(f"\n {name} \n"),
            _table([[" Input ", " Output "], [f" {indim} ", f" {outdim} "]]),
        ]
    )


# tree_**


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

    def recurse(tree, path=(), frozen_state=None):

        nonlocal FMT, COUNT, SIZE

        if is_treeclass(tree):

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
                                f"{k}={_format_node_repr(v)}"
                                for k, v in cur_node.__tree_fields__[0].items()
                            ]
                        )
                        if is_treeclass(cur_node)
                        else f"{fi.name}={_format_node_repr(cur_node)}"
                    )

                    shape_str = (
                        f"{_format_node_repr(indim_shape[i])}\n{_format_node_repr(outdim_shape[i])}"
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

    def recurse(tree, path=(), frozen_state=None):

        nonlocal ROWS, COUNT, SIZE

        if is_treeclass(tree):

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
                    frozen_str = "\n(frozen)" if frozen_state else ""
                    name_str = f"{fi.name}{frozen_str}"
                    type_str = "/".join(path + (cur_node.__class__.__name__,))
                    count_str = _format_count(count, True)
                    size_str = _format_size(size, True)
                    config_str = (
                        "\n".join(
                            [
                                f"{k}={_format_node_repr(v)}"
                                for k, v in cur_node.__tree_fields__[0].items()
                            ]
                        )
                        if is_treeclass(cur_node)
                        else f"{fi.name}={_format_node_repr(cur_node)}"
                    )

                    shape_str = (
                        f"{_format_node_repr(indim_shape[i])}\n{_format_node_repr(outdim_shape[i])}"
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
                _format_node_repr(shapes[0]) if array is not None else None,
                _format_node_repr(shapes[1]) if array is not None else None,
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
                    level_nodes += [_vbox(f"{fi.name}={_format_node_repr(cur_node)}")]

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
            is_static = (
                "static" in field_item.metadata and field_item.metadata["static"]
            )
            mark = "x" if is_static else ("#" if frozen_state else "─")
            is_last_field = node_index == 1

            FMT += "\n"
            FMT += "".join(
                [(("│" if lvl > 1 else "") + "\t") for lvl in parent_level_count]
            )

            FMT += f"└{mark}─ " if is_last_field else f"├{mark}─ "
            FMT += f"{field_item.name}={_format_node_diagram(node_item)}"

        recurse(node_item, parent_level_count + [1], frozen_state)

    @recurse_field.register(pytreeclass.src.tree_base.treeBase)
    def _(field_item, node_item, frozen_state, parent_level_count, node_index):
        nonlocal FMT
        assert is_treeclass(node_item)

        if field_item.repr:
            frozen_state = node_item.frozen
            is_static = (
                "static" in field_item.metadata and field_item.metadata["static"]
            )
            mark = "x" if is_static else ("#" if frozen_state else "─")
            layer_class_name = node_item.__class__.__name__

            is_last_field = node_index == 1

            FMT += "\n" + "".join(
                [(("│" if lvl > 1 else "") + "\t") for lvl in parent_level_count]
            )

            FMT += (
                f"└{mark}─ " if is_last_field else f"├{mark}─ "
            ) + f"{field_item.name}={layer_class_name}"

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
            FMT += "\n" + "\t" * depth
            FMT += (
                f"{field_item.name}={format_width(_format_node_repr(node_item,depth))}"
            )
            FMT += "" if is_last_field else ","

        recurse(node_item, depth, frozen_state)

    @recurse_field.register(pytreeclass.src.tree_base.treeBase)
    def _(field_item, node_item, depth, frozen_state, is_last_field):
        """format treeclass field"""
        nonlocal FMT
        assert is_treeclass(node_item)
        if field_item.repr:
            FMT += "\n" + "\t" * depth
            layer_class_name = f"{node_item.__class__.__name__}"
            frozen_str = "#" if node_item.frozen else ""
            FMT += f"{frozen_str}{field_item.name}={layer_class_name}" + "("
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
            FMT += "\n" + "\t" * depth
            FMT += (
                f"{field_item.name}={format_width(_format_node_str(node_item,depth))}"
            )
            FMT += "" if is_last_field else ","

        recurse(node_item, depth, frozen_state)

    @recurse_field.register(pytreeclass.src.tree_base.treeBase)
    def _(field_item, node_item, depth, frozen_state, is_last_field):
        """format treeclass field"""
        nonlocal FMT
        assert is_treeclass(node_item)

        if field_item.repr:
            FMT += "\n" + "\t" * depth
            layer_class_name = f"{node_item.__class__.__name__}"
            frozen_str = "#" if node_item.frozen else ""
            FMT += f"{frozen_str}{field_item.name}={layer_class_name}" + "("
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

    def recurse(tree, cur_depth, prev_id):

        nonlocal FMT

        if is_treeclass(tree):

            for i, fi in enumerate(tree.__dataclass_fields__.values()):
                if fi.repr:
                    cur_node = tree.__dict__[fi.name]
                    cur_order = i
                    FMT += "\n"

                    if is_treeclass(cur_node):
                        layer_class_name = cur_node.__class__.__name__
                        cur = (cur_depth, cur_order)
                        cur_id = node_id((*cur, prev_id))
                        FMT += f"\tid{prev_id} --> id{cur_id}({fi.name}\\n{layer_class_name})"
                        recurse(cur_node, cur_depth + 1, cur_id)

                    else:
                        cur = (cur_depth, cur_order)
                        cur_id = node_id((*cur, prev_id))
                        is_static = "static" in fi.metadata and fi.metadata["static"]
                        connector = (
                            "--x" if is_static else ("-.-" if tree.frozen else "---")
                        )
                        FMT += f'\tid{prev_id} {connector} id{cur_id}["{fi.name}\\n{_format_node_repr(cur_node)}"]'
                        recurse(cur_node, cur_depth + 1, cur_id)

                elif not is_treeclass(tree):
                    recurse(cur_node, cur_depth + 1, cur_id)

    cur_id = node_id((0, 0, -1, 0))
    FMT = f"flowchart LR\n\tid{cur_id}[{tree.__class__.__name__}]"
    recurse(tree, 1, cur_id)
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
