from __future__ import annotations

import inspect
import math
from types import FunctionType
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jaxlib

from pytreeclass.src.decorator_util import dispatch

PyTree = Any


# Node formatting


def _func_repr(func):
    args, varargs, varkw, _, kwonlyargs, _, _ = inspect.getfullargspec(func)
    args = (",".join(args)) if len(args) > 0 else ""
    varargs = ("*" + varargs) if varargs is not None else ""
    kwonlyargs = (",".join(kwonlyargs)) if len(kwonlyargs) > 0 else ""
    varkw = ("**" + varkw) if varkw is not None else ""
    name = "Lambda" if (func.__name__ == "<lambda>") else func.__name__
    return (
        f"{name}("
        + ",".join(item for item in [args, varargs, kwonlyargs, varkw] if item != "")
        + ")"
    )


def _jax_numpy_repr(node):
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


def _format_width(string, width=50):
    """strip newline/tab characters if less than max width"""
    stripped_string = string.replace("\n", "").replace("\t", "")
    children_length = len(stripped_string)
    return string if children_length > width else stripped_string


def _format_node_repr(node, depth):
    @dispatch(argnum=0)
    def __format_node_repr(node, depth):
        return ("\n" + "\t" * (depth)).join(f"{node!r}".split("\n"))

    @__format_node_repr.register(jaxlib.xla_extension.CompiledFunction)
    @__format_node_repr.register(jax._src.custom_derivatives.custom_jvp)
    @__format_node_repr.register(FunctionType)
    def _(node, *args, **kwargs):
        return _func_repr(node)

    @__format_node_repr.register(jnp.ndarray)
    @__format_node_repr.register(jax.ShapeDtypeStruct)
    def _(node, *args, **kwargs):
        return _jax_numpy_repr(node)

    @__format_node_repr.register(list)
    def _(node, depth):
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{_format_width(_format_node_repr(v,depth=depth+1))}" for v in node
        )
        return "[\n" + "\t" * (depth + 1) + string + "\n" + "\t" * (depth) + "]"

    @__format_node_repr.register(tuple)
    def _(node, depth):
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{_format_width(_format_node_repr(v,depth=depth+1))}" for v in node
        )
        return "(\n" + "\t" * (depth + 1) + string + "\n" + "\t" * (depth) + ")"

    @__format_node_repr.register(dict)
    def _(node, depth):
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{k}:{_format_node_repr(v,depth=depth+1)}"
            if "\n" not in f"{v!s}"
            else f"{k}:"
            + "\n"
            + "\t" * (depth + 1)
            + f"{_format_width(_format_node_repr(v,depth=depth+1))}"
            for k, v in node.items()
        )
        return "{\n" + "\t" * (depth + 1) + string + "\n" + "\t" * (depth) + "}"

    return __format_node_repr(node, depth)


def _format_node_str(node, depth):
    @dispatch(argnum=0)
    def __format_node_str(node, depth):
        return ("\n" + "\t" * (depth)).join(f"{node!s}".split("\n"))

    @__format_node_str.register(jaxlib.xla_extension.CompiledFunction)
    @__format_node_str.register(jax._src.custom_derivatives.custom_jvp)
    @__format_node_str.register(FunctionType)
    def _(node, *args, **kwargs):
        return _func_repr(node)

    @__format_node_str.register(list)
    def _(node, depth):
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{_format_width(_format_node_str(v,depth=depth+1))}" for v in node
        )
        return "[\n" + "\t" * (depth + 1) + string + "\n" + "\t" * (depth) + "]"

    @__format_node_str.register(tuple)
    def _(node, depth):
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{_format_width(_format_node_str(v,depth=depth+1))}" for v in node
        )
        return "(\n" + "\t" * (depth + 1) + string + "\n" + "\t" * (depth) + ")"

    @__format_node_str.register(dict)
    def _(node, depth):
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{k}:{_format_node_str(v,depth=depth+1)}"
            if "\n" not in f"{v!s}"
            else f"{k}:"
            + "\n"
            + "\t" * (depth + 1)
            + f"{_format_width(_format_node_str(v,depth=depth+1))}"
            for k, v in node.items()
        )
        return "{\n" + "\t" * (depth + 1) + string + "\n" + "\t" * (depth) + "}"

    return _format_width(__format_node_str(node, depth))


def _format_node_diagram(node, *args, **kwargs):
    @dispatch(argnum=0)
    def __format_node_diagram(node, *args, **kwargs):
        return f"{node!r}"

    @__format_node_diagram.register(jaxlib.xla_extension.CompiledFunction)
    @__format_node_diagram.register(jax._src.custom_derivatives.custom_jvp)
    @__format_node_diagram.register(FunctionType)
    def _(node, *args, **kwargs):
        return _func_repr(node)

    @__format_node_diagram.register(jnp.ndarray)
    @__format_node_diagram.register(jax.ShapeDtypeStruct)
    def _(node, *args, **kwargs):
        return _jax_numpy_repr(node)

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
