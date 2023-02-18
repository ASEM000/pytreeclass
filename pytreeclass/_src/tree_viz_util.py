from __future__ import annotations

import dataclasses as dc
import math
import sys
from itertools import chain
from typing import Any, NamedTuple, Sequence

import numpy as np
from jax._src.tree_util import _registry

from pytreeclass._src.tree_freeze import is_frozen, unfreeze

PyTree = Any


def _format_width(string, width=60):
    """strip newline/tab characters if less than max width"""
    children_length = len(string) - string.count("\n") - string.count("\t")
    if children_length > width:
        return string
    return string.replace("\n", "").replace("\t", "")


def _format_size(node_size, newline=False):
    """return formatted size from inexact(exact) complex number

    Examples:
        >>> _format_size(1024)
        '1.00KB'
        >>> _format_size(1024**2)
        '1.00MB'
    """
    mark = "\n" if newline else ""
    order_kw = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]

    if isinstance(node_size, complex):
        # define order of magnitude
        real_size_order = int(math.log(max(node_size.real, 1), 1024))
        imag_size_order = int(math.log(max(node_size.imag, 1), 1024))
        fmt = f"{(node_size.real)/(1024**real_size_order):.2f}{order_kw[real_size_order]}{mark}"
        fmt += f"({(node_size.imag)/(1024**imag_size_order):.2f}{order_kw[imag_size_order]})"
        return fmt

    if isinstance(node_size, (float, int)):
        size_order = int(math.log(node_size, 1024)) if node_size > 0 else 0
        return f"{(node_size)/(1024**size_order):.2f}{order_kw[size_order]}"

    raise TypeError(f"node_size must be int or float, got {type(node_size)}")


def _format_count(node_count, newline=False):
    """return formatted count from inexact(exact) complex number

    Examples:
        >>> _format_count(1024)
        '1,024'

        >>> _format_count(1024**2)
        '1,048,576'
    """

    mark = "\n" if newline else ""

    if isinstance(node_count, complex):
        return f"{int(node_count.real):,}{mark}({int(node_count.imag):,})"

    if isinstance(node_count, (float, int)):
        return f"{int(node_count):,}"

    raise TypeError(f"node_count must be int or float, got {type(node_count)}")


def is_children_frozen(tree):
    """assert if a dataclass is static"""
    if dc.is_dataclass(tree):
        fields = dc.fields(tree)
        if len(fields) > 0:
            if all(is_frozen(f) for f in fields):
                return True
            if all(is_frozen(getattr(tree, f.name)) for f in fields):
                return True

    return False


def _mermaid_marker(field: dc.Field, node: Any, default: str = "--") -> str:
    """return the suitable marker given the field and node item

    Args:
        field (Field): field item of the pytree node
        node (Any): node item
        default (str, optional): default marker. Defaults to "".

    Returns:
        str: marker character.
    """
    if is_frozen(field) or is_children_frozen(node):
        return "--x"
    return default


def _marker(field: dc.Field, node: Any, default: str = "") -> str:
    """return the suitable marker given the field and node item"""
    # if all your children are frozen, the you are frozen
    if is_frozen(field) or is_children_frozen(node):
        return "#"

    return default


class NodeInfo(NamedTuple):
    node: Any  # node item
    names: list[str]  # name of nodes in each level
    types: list[type]  # types of nodes in each level
    index: list[int]  # reversed index of the node in each level. index=0 is last child
    frozen: bool = False  # if the node is frozen
    repr: bool = True  # if the node is reprable


def _tree_trace(tree: PyTree, depth=float("inf")) -> list[NodeInfo]:
    """trace the tree and return a list of NodeInfo for a given depth"""

    def yield_one_level_leaves(info: NodeInfo, depth: int):
        if depth < 1:
            yield info
            return

        if type(info.node) not in _registry:
            # the tree is not a registered JAX pytree
            yield info
            return

        # JAX registered pytree case
        # check for python datastructure that has named paths first
        # otherwise assign `leaf_` for other `JAXable` objects
        handler = _registry.get(type(info.node))
        leaves = list(handler.to_iter(info.node)[0])

        # frozen rules: parent or the node is frozen and associated node field is frozen -if the node is a dataclass-
        # repr rules: parent is repr and the associated node field is repr -if the node is a dataclass-
        if isinstance(info.node, tuple) and hasattr(info.node, "_fields"):
            names = (f"{key}" for key in info.node._fields)
            reprs = (info.repr,) * len(info.node)
            frozen = (is_frozen(leaf) or info.frozen for leaf in leaves)
        elif isinstance(info.node, dict):
            names = (f"{key}" for key in info.node)
            reprs = (info.repr,) * len(info.node)
            frozen = (is_frozen(leaf) or info.frozen for leaf in leaves)
        elif isinstance(info.node, (list, tuple)):
            names = (slice(i, i + 1) for i in range(len(info.node)))
            reprs = (info.repr,) * len(info.node)
            frozen = (is_frozen(leaf) or info.frozen for leaf in leaves)
        elif dc.is_dataclass(info.node):
            names = (f"{f.name}" for f in dc.fields(info.node))
            reprs = (info.repr and f.repr for f in dc.fields(info.node))
            lfs = zip(leaves, dc.fields(info.node))
            frozen = (is_frozen(leaf) or info.frozen or is_frozen(field) for leaf, field in lfs)  # fmt: skip
        else:
            names = (f"leaf_{i}" for i in range(len(leaves)))
            reprs = (info.repr,) * len(leaves)
            frozen = (is_frozen(leaf) or info.frozen for leaf in leaves)

        for i, (leaf, name, repr, frozen) in enumerate(
            zip(leaves, names, reprs, frozen)
        ):
            names = info.names + [name]
            types = info.types + [type(leaf)]
            # reversed index of the leaf, index=0 is the last leaf
            index = info.index + [len(leaves) - i - 1]
            sub_info = NodeInfo(unfreeze(leaf), names, types, index, frozen, repr)
            yield from (yield_one_level_leaves(sub_info, depth - 1))

    # in general, frozen nodes will yield no leaves if `jtu.tree_leaves` is applied,
    # however in this function we wish to trace every leaf, so we unfreeze the tree first.
    info = NodeInfo(
        node=unfreeze(tree),
        names=[],
        types=list(),
        index=list(),
        frozen=is_frozen(tree),
        repr=True,
    )

    return list(yield_one_level_leaves(info, depth))


def _calculate_node_info_stats(info: NodeInfo) -> tuple[int | complex, int | complex]:
    # calcuate some stats of a single subtree defined by the `NodeInfo` objects
    # for each subtree, we will calculate the types distribution and their size
    # stats = defaultdict(lambda: [0, 0])
    if not isinstance(info, NodeInfo):
        raise TypeError(f"Expected `NodeInfo` object, but got {type(info)}")

    count = size = 0

    for sub_info in _tree_trace(info.node, float("inf")):
        # get all the leaves of the subtree
        node = sub_info.node

        # array count is the product of the shape. if the node is not an array, then the count is 1
        count_ = int(np.array(node.shape).prod()) if hasattr(node, "shape") else 1
        # if the node is frozen, then the count is imaginary otherwise it is real
        count_ = complex(0, count_) if (sub_info.frozen or info.frozen) else count_
        count += count_
        size_ = node.nbytes if hasattr(node, "nbytes") else sys.getsizeof(node)
        size_ = complex(0, size_) if (sub_info.frozen or info.frozen) else size_
        size += size_

    return (count, size)


# table printing


def _hbox(*text) -> str:
    """Create horizontally stacked text boxes

    Examples:
        >>> _hbox("a","b")
        ┌─┬─┐
        │a│b│
        └─┴─┘
    """
    boxes = list(map(_vbox, text))
    boxes = [(box).split("\n") for box in boxes]
    max_col_height = max([len(b) for b in boxes])
    boxes = [b + [" " * len(b[0])] * (max_col_height - len(b)) for b in boxes]
    return "\n".join([_resolve_line(line) for line in zip(*boxes)])


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
        max(chain.from_iterable([[len(t) for t in item.split("\n")] for item in text]))
        + 0
    )

    top = f"┌{'─'*max_width}┐"
    line = f"├{'─'*max_width}┤"
    side = [
        "\n".join([f"│{t}{' '*(max_width-len(t))}│" for t in item.split("\n")])
        for item in text
    ]
    btm = f"└{'─'*max_width}┘"

    fmt = ""

    for i, s in enumerate(side):

        if i == 0:
            fmt += f"{top}\n{s}\n{line if len(side)>1 else btm}"

        elif i == len(side) - 1:
            fmt += f"\n{s}\n{btm}"

        else:
            fmt += f"\n{s}\n{line}"

    return fmt


def _hstack(*boxes):
    """Create horizontally stacked text boxes

    Examples:
        >>> print(_hstack(_hbox("a"),_vbox("b","c")))
        ┌─┬─┐
        │a│b│
        └─┼─┤
          │c│
          └─┘
    """
    boxes = [(box).split("\n") for box in boxes]
    max_col_height = max([len(b) for b in boxes])
    # expand height of each col before merging
    boxes = [b + [" " * len(b[0])] * (max_col_height - len(b)) for b in boxes]
    FMT = ""

    _cells = tuple(zip(*boxes))

    for i, line in enumerate(_cells):
        FMT += _resolve_line(line) + ("\n" if i != (len(_cells) - 1) else "")

    return FMT


def _resolve_line(cols: Sequence[str]) -> str:
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


def _table(lines: Sequence[str]) -> str:
    """create a table with self aligning rows and cols

    Args:
        lines (Sequence[str,...]): list of lists of cols values

    Returns:
        str: box string

    Example:
        >>> col1 = ['1\n','2']
        >>> col2 = ['3','4000']
        >>> print(_table([col1,col2]))
        ┌─┬────────┐
        │1│3       │
        │ │        │
        ├─┼────────┤
        │2│40000000│
        └─┴────────┘
    """
    for i, _cells in enumerate(zip(*lines)):
        max_cell_height = max(map(lambda x: x.count("\n"), _cells))
        for j in range(len(_cells)):
            lines[j][i] += "\n" * (max_cell_height - lines[j][i].count("\n"))

    return _hstack(*(_vbox(*col) for col in lines))
