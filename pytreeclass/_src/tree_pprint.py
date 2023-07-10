# Copyright 2023 PyTreeClass authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for pretty printing pytrees."""

from __future__ import annotations

import dataclasses as dc
import functools as ft
import inspect
import math
from itertools import chain
from types import FunctionType
from typing import Any, Callable, Iterable, Literal

import jax
import jax.tree_util as jtu
import numpy as np
from jax import custom_jvp
from jax.util import unzip2
from typing_extensions import TypedDict, Unpack

from pytreeclass._src.tree_util import (
    IsLeafType,
    Node,
    construct_tree,
    tree_leaves_with_trace,
    tree_map_with_trace,
)


class PPSpec(TypedDict):
    indent: int
    kind: Literal["REPR", "STR"]
    width: int
    depth: int | float
    seen: set[int]


PyTree = Any
PP = Callable[[Any, Unpack[PPSpec]], str]
from_iterable = chain.from_iterable


@ft.singledispatch
def pp_dispatcher(node: Any, **spec: Unpack[PPSpec]) -> str:
    """Register a new or override an existing pretty printer by type using."""
    return general_pp(node, **spec)


def dataclass_pp(node: Any, **spec: Unpack[PPSpec]) -> str:
    name = type(node).__name__

    kvs = ((f.name, vars(node)[f.name]) for f in dc.fields(node) if f.repr)
    return name + "(" + pps(kvs, pp=attr_value_pp, **spec) + ")"


def general_pp(node: Any, **spec: Unpack[PPSpec]) -> str:
    # ducktyping and other fallbacks that are not covered by singledispatch

    if dc.is_dataclass(node):
        return dataclass_pp(node, **spec)

    text = f"{node!r}" if spec["kind"] == "REPR" else f"{node!s}"

    if "\n" not in text:
        return text

    return ("\n" + "\t" * (spec["indent"])).join(text.split("\n"))


def pp(node: Any, **spec: Unpack[PPSpec]) -> str:
    if spec["depth"] < 0:
        return "..."

    if (node_id := id(node)) in spec["seen"]:
        # useful to avoid infinite recursion in cyclic references
        # e.g. (a:=[1,2,3];a.append(a))
        return f"<cyclic reference to {node_id}>"

    return format_width(pp_dispatcher(node, **spec), width=spec["width"])


def pps(xs: Iterable[Any], pp: PP, **spec: Unpack[PPSpec]) -> str:
    if spec["depth"] == 0:
        return "..."

    spec["indent"] += 1
    spec["depth"] -= 1
    spec["seen"].add(id(xs))  # avoid infinite recursion in cyclic references

    text = (
        "\n"
        + "\t" * spec["indent"]
        + (", \n" + "\t" * spec["indent"]).join(pp(x, **spec) for x in xs)
        + "\n"
        + "\t" * (spec["indent"] - 1)
    )

    return format_width(text, width=spec["width"])


def key_value_pp(x: tuple[str, Any], **spec: Unpack[PPSpec]) -> str:
    return f"{x[0]}:{pp(x[1], **spec)}"


def attr_value_pp(x: tuple[str, Any], **spec: Unpack[PPSpec]) -> str:
    return f"{x[0]}={pp(x[1], **spec)}"


@pp_dispatcher.register(jax.ShapeDtypeStruct)
def shape_dtype_pp(node: Any, **spec: Unpack[PPSpec]) -> str:
    """Pretty print a node with dtype and shape."""
    shape = f"{node.shape}".replace(",", "")
    shape = shape.replace("(", "[")
    shape = shape.replace(")", "]")
    shape = shape.replace(" ", ",")
    dtype = f"{node.dtype}".replace("int", "i")
    dtype = dtype.replace("float", "f")
    dtype = dtype.replace("complex", "c")
    return dtype + shape


@pp_dispatcher.register(np.ndarray)
@pp_dispatcher.register(jax.Array)
def numpy_pp(node: np.ndarray | jax.Array, **spec: Unpack[PPSpec]) -> str:
    """Replace np.ndarray repr with short hand notation for type and shape."""
    if spec["kind"] == "STR":
        return general_pp(node, **spec)

    base = shape_dtype_pp(node, **spec)

    if not issubclass(node.dtype.type, (np.integer, np.floating)) or node.size == 0:
        return base

    # Extended repr for numpy array, with extended information
    # this part of the function is inspired by
    # lovely-jax https://github.com/xl0/lovely-jax

    # handle interval
    low, high = np.min(node), np.max(node)
    interval = "(" if math.isinf(low) else "["
    is_integer = issubclass(node.dtype.type, np.integer)
    interval += f"{low},{high}" if is_integer else f"{low:.2f},{high:.2f}"
    interval += ")" if math.isinf(high) else "]"  # resolve closed/open interval
    interval = interval.replace("inf", "∞")  # replace inf with infinity symbol
    # handle mean and std
    mean, std = f"{np.mean(node):.2f}", f"{np.std(node):.2f}"

    return f"{base}(μ={mean}, σ={std}, ∈{interval})"


@pp_dispatcher.register(FunctionType)
@pp_dispatcher.register(custom_jvp)
def func_pp(func: Callable, **spec: Unpack[PPSpec]) -> str:
    # Pretty print function
    # Example:
    #     >>> def example(a: int, b=1, *c, d, e=2, **f) -> str:
    #         ...
    #     >>> func_pp(example)
    #     "example(a, b, *c, d, e, **f)"
    del spec
    args, varargs, varkw, _, kwonlyargs, _, _ = inspect.getfullargspec(func)
    args = (", ".join(args)) if len(args) > 0 else ""
    varargs = ("*" + varargs) if varargs is not None else ""
    kwonlyargs = (", ".join(kwonlyargs)) if len(kwonlyargs) > 0 else ""
    varkw = ("**" + varkw) if varkw is not None else ""
    name = getattr(func, "__name__", "")
    text = f"{name}("
    text += ", ".join(item for item in [args, varargs, kwonlyargs, varkw] if item != "")
    text += ")"
    return text


@pp_dispatcher.register(ft.partial)
def partial_pp(node: ft.partial, **spec: Unpack[PPSpec]) -> str:
    return f"Partial({func_pp(node.func, **spec)})"


@pp_dispatcher.register(list)
def list_pp(node: list, **spec: Unpack[PPSpec]) -> str:
    return "[" + pps(node, pp=pp, **spec) + "]"


@pp_dispatcher.register(tuple)
def tuple_pp(node: tuple, **spec: Unpack[PPSpec]) -> str:
    if not hasattr(node, "_fields"):
        return "(" + pps(node, pp=pp, **spec) + ")"
    name = type(node).__name__
    kvs = node._asdict().items()
    return name + "(" + pps(kvs, pp=attr_value_pp, **spec) + ")"


@pp_dispatcher.register(set)
def set_pp(node: set, **spec: Unpack[PPSpec]) -> str:
    return "{" + pps(node, pp=pp, **spec) + "}"


@pp_dispatcher.register(dict)
def dict_pp(node: dict, **spec: Unpack[PPSpec]) -> str:
    return "{" + pps(node.items(), pp=key_value_pp, **spec) + "}"


def tree_repr(
    tree: PyTree,
    *,
    width: int = 80,
    tabwidth: int = 2,
    depth: int | float = float("inf"),
) -> str:
    """Prertty print arbitrary PyTrees `__repr__`.

    Args:
        tree: PyTree
        width: max width of the repr string
        tabwidth: tab width of the repr string
        depth: max depth of the repr string

    Example:
        >>> import pytreeclass as pytc
        >>> import jax.numpy as jnp
        >>> tree = {'a' : 1, 'b' : [2, 3], 'c' : {'d' : 4, 'e' : 5} , 'f' : jnp.array([6, 7])}

        >>> print(pytc.tree_repr(tree, depth=0))
        {...}

        >>> print(pytc.tree_repr(tree, depth=1))
        {a:1, b:[...], c:{...}, f:i32[2](μ=6.50, σ=0.50, ∈[6,7])}

        >>> print(pytc.tree_repr(tree, depth=2))
        {a:1, b:[2, 3], c:{d:4, e:5}, f:i32[2](μ=6.50, σ=0.50, ∈[6,7])}
    """
    text = pp(tree, indent=0, kind="REPR", width=width, depth=depth, seen=set())
    return text.expandtabs(tabwidth)


def tree_str(
    tree: PyTree,
    *,
    width: int = 80,
    tabwidth: int = 2,
    depth: int | float = float("inf"),
) -> str:
    """Prertty print arbitrary PyTrees `__str__`.

    Args:
        tree: PyTree
        width: max width of the str string
        tabwidth: tab width of the repr string
        depth: max depth of the repr string
    Example:
        >>> import pytreeclass as pytc
        >>> import jax.numpy as jnp
        >>> tree = {'a' : 1, 'b' : [2, 3], 'c' : {'d' : 4, 'e' : 5} , 'f' : jnp.array([6, 7])}

        >>> print(pytc.tree_str(tree, depth=1))
        {a:1, b:[...], c:{...}, f:[6 7]}

        >>> print(pytc.tree_str(tree, depth=2))
        {a:1, b:[2, 3], c:{d:4, e:5}, f:[6 7]}
    """
    text = pp(tree, indent=0, kind="STR", width=width, depth=depth, seen=set())
    return text.expandtabs(tabwidth)


def _is_trace_leaf_depth_factory(depth: int | float):
    # generate `is_trace_leaf` function to stop tracing at a certain `depth`
    # in essence, depth is the length of the trace entry
    def is_trace_leaf(trace) -> bool:
        # trace is a tuple of (names, leaves, tracers, aux_data)
        # done like this to ensure 4-tuple unpacking
        keys, _ = trace
        # stop tracing if depth is reached
        return False if depth is None else (depth <= len(keys))

    return is_trace_leaf


def tree_indent(
    tree: Any,
    *,
    depth: int | float = float("inf"),
    is_leaf: IsLeafType = None,
    tabwidth: int | None = 4,
):
    """Returns a string representation of the tree with indentation.

    Args:
        tree: The tree to be printed.
        depth: The maximum depth of the tree to be printed.
        is_leaf: A function that takes a node and returns True if it is a leaf node.
        tabwidth: The number of spaces per indentation level. if `None`
            then tabs are not expanded.

    Example:
        >>> import pytreeclass as pytc
        >>> tree = [1, [2, 3], [4, [5, 6]]]
        >>> print(pytc.tree_indent(tree))
        list
            [0]=1
            [1]:list
                [0]=2
                [1]=3
            [2]:list
                [0]=4
                [1]:list
                    [0]=5
                    [1]=6

    """
    smark = (" \t")[:tabwidth]  # space mark

    def step(node: Node, depth: int = 0) -> str:
        indent = smark * depth

        if (len(node.children)) == 0:
            ppspec = dict(indent=0, kind="REPR", width=80, depth=0, seen=set())
            text = f"{indent}"
            (key, _), value = node.data
            text += f"{key}=" if key is not None else ""
            text += pp(value, **ppspec)
            return text + "\n"

        (key, type), _ = node.data
        text = f"{indent}"
        text += f"{key}:" if key is not None else ""
        text += f"{type.__name__}\n"

        for child in node:
            text += step(child, depth=depth + 1)
        return text

    root = construct_tree(
        tree,
        is_leaf=is_leaf,
        is_trace_leaf=_is_trace_leaf_depth_factory(depth),
    )
    text = step(root)
    return (text if tabwidth is None else text.expandtabs(tabwidth)).rstrip()


def tree_diagram(
    tree: Any,
    *,
    depth: int | float = float("inf"),
    is_leaf: IsLeafType = None,
    tabwidth: int = 4,
):
    """Pretty print arbitrary PyTrees tree with tree structure diagram.

    Args:
        tree: PyTree
        depth: depth of the tree to print. default is max depth
        is_leaf: function to determine if a node is a leaf. default is None.
        tabwidth: tab width of the repr string. default is 4.

    Example:
        >>> import pytreeclass as pytc
        >>> class A(pytc.TreeClass):
        ...        x: int = 10
        ...        y: int = (20,30)
        ...        z: int = 40

        >>> class B(pytc.TreeClass):
        ...     a: int = 10
        ...     b: tuple = (20,30, A())

        >>> print(pytc.tree_diagram(B(), depth=0))
        B(...)

        >>> print(pytc.tree_diagram(B(), depth=1))
        B
        ├── .a=10
        └── .b=(...)


        >>> print(pytc.tree_diagram(B(), depth=2))
        B
        ├── .a=10
        └── .b:tuple
            ├── [0]=20
            ├── [1]=30
            └── [2]=A(...)
    """
    vmark = ("│\t")[:tabwidth]  # vertical mark
    lmark = ("└" + "─" * (tabwidth - 2) + (" \t"))[:tabwidth]  # last mark
    cmark = ("├" + "─" * (tabwidth - 2) + (" \t"))[:tabwidth]  # connector mark
    smark = (" \t")[:tabwidth]  # space mark

    def step(
        node: Node,
        depth: int = 0,
        is_last: bool = False,
        is_lasts: tuple[bool, ...] = (),
    ) -> str:
        indent = "".join(smark if is_last else vmark for is_last in is_lasts[:-1])
        branch = (lmark if is_last else cmark) if depth > 0 else ""

        if (child_count := len(node.children)) == 0:
            ppspec = dict(indent=0, kind="REPR", width=80, depth=0, seen=set())
            (key, _), value = node.data
            text = f"{indent}"
            text += f"{branch}{key}=" if key is not None else ""
            text += pp(value, **ppspec)
            return text + "\n"

        (key, type), _ = node.data

        text = f"{indent}{branch}"
        text += f"{key}:" if key is not None else ""
        text += f"{type.__name__}\n"

        for i, child in enumerate(node.children.values()):
            text += step(
                child,
                depth=depth + 1,
                is_last=(i == child_count - 1),
                is_lasts=is_lasts + (i == child_count - 1,),
            )
        return text

    root = construct_tree(
        tree,
        is_leaf=is_leaf,
        is_trace_leaf=_is_trace_leaf_depth_factory(depth),
    )
    text = step(root, is_last=len(root.children) == 1)
    return (text if tabwidth is None else text.expandtabs(tabwidth)).rstrip()


def tree_mermaid(
    tree: PyTree,
    depth: int | float = float("inf"),
    is_leaf: IsLeafType = None,
    tabwidth: int | None = 4,
) -> str:
    """Generate a mermaid diagram syntax for arbitrary PyTrees.

    Args:
        tree: PyTree
        depth: depth of the tree to print. default is max depth
        is_leaf: function to determine if a node is a leaf. default is None
        tabwidth: tab width of the repr string. default is 4.
    """

    def step(node: Node, depth: int = 0) -> str:
        if len(node.children) == 0:
            ppspec = dict(indent=0, kind="REPR", width=80, depth=0, seen=set())
            (key, _), value = node.data
            text = f"{key}=" if key is not None else ""
            text += pp(value, **ppspec)
            text = "<b>" + text + "</b>"
            return f'\tid{id(node.parent)} --- id{id(node)}("{text}")\n'

        (key, type), _ = node.data
        text = f"{key}:" if key is not None else ""
        text += f"{type.__name__}"
        text = "<b>" + text + "</b>"

        if node.parent is None:
            text = f'\tid{id(node)}("{text}")\n'
        else:
            text = f'\tid{id(node.parent)} --- id{id(node)}("{text}")\n'

        for child in node.children.values():
            text += step(child, depth=depth + 1)
        return text

    root = construct_tree(
        tree,
        is_leaf=is_leaf,
        is_trace_leaf=_is_trace_leaf_depth_factory(depth),
    )
    text = "flowchart LR\n" + step(root)

    return (text.expandtabs(tabwidth) if tabwidth is not None else text).rstrip()


def format_width(string, width=60):
    """Strip newline/tab characters if less than max width."""
    children_length = len(string) - string.count("\n") - string.count("\t")
    if children_length > width:
        return string
    return string.replace("\n", "").replace("\t", "")


# table printing


def _hbox(*text) -> str:
    # Create horizontally stacked text boxes
    # Examples:
    #     >>> _hbox("a","b")
    #     ┌─┬─┐
    #     │a│b│
    #     └─┴─┘

    boxes = list(map(_vbox, text))
    boxes = [(box).split("\n") for box in boxes]
    max_col_height = max([len(b) for b in boxes])
    boxes = [b + [" " * len(b[0])] * (max_col_height - len(b)) for b in boxes]
    return "\n".join([_resolve_line(line) for line in zip(*boxes)])


def _vbox(*text) -> str:
    # Create vertically stacked text boxes
    # Example:
    #     >>> _vbox("a","b")
    #     ┌───┐
    #     │a  │
    #     ├───┤
    #     │b  │
    #     └───┘

    #     >>> _vbox("a","","a")
    #     ┌───┐
    #     │a  │
    #     ├───┤
    #     │   │
    #     ├───┤
    #     │a  │
    #     └───┘

    max_width = (
        max(from_iterable([[len(t) for t in item.split("\n")] for item in text])) + 0
    )

    top = f"┌{'─'*max_width}┐"
    line = f"├{'─'*max_width}┤"
    side = [
        "\n".join([f"│{t}{' '*(max_width-len(t))}│" for t in item.split("\n")])
        for item in text
    ]

    btm = f"└{'─'*max_width}┘"

    text = ""

    for i, s in enumerate(side):
        if i == 0:
            text += f"{top}\n{s}\n{line if len(side)>1 else btm}"

        elif i == len(side) - 1:
            text += f"\n{s}\n{btm}"

        else:
            text += f"\n{s}\n{line}"

    return text


def _hstack(*boxes):
    # Create horizontally stacked text boxes
    # Example:
    #     >>> print(_hstack(_hbox("a"),_vbox("b","c")))
    #     ┌─┬─┐
    #     │a│b│
    #     └─┼─┤
    #       │c│
    #       └─┘

    boxes = [(box).split("\n") for box in boxes]
    max_col_height = max([len(b) for b in boxes])
    # expand height of each col before merging
    boxes = [b + [" " * len(b[0])] * (max_col_height - len(b)) for b in boxes]
    text = ""

    _cells = tuple(zip(*boxes))

    for i, line in enumerate(_cells):
        text += _resolve_line(line) + ("\n" if i != (len(_cells) - 1) else "")

    return text


def _resolve_line(cols: list[str]) -> str:
    # Combine columns of single line by merging their borders

    # Args:
    #     cols (Sequence[str,...]): Sequence of single line column string

    # Returns:
    #     str: resolved column string

    # Example:
    #     >>> _resolve_line(['ab','b│','│c'])
    #     'abb│c'

    #     >>> _resolve_line(['ab','b┐','┌c'])
    #     'abb┬c'

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

        elif cols[index][-1] == " ":
            cols[index].pop()

        elif cols[index][-1] in alpha and cols[index + 1][0] in [*alpha, " "]:
            cols[index + 1].pop(0)

    return "".join(map(lambda x: "".join(x), cols))


def _table(rows: list[list[str]], transpose: bool = False) -> str:
    # Create a table with self aligning rows and cols

    # Args:
    #     rows: list of lists of row values
    #     transpose: transpose the table. i.e. rows become cols and cols become rows

    # Returns:
    #     str: box string

    # Example:
    #     >>> col1 = ['1\n','2']
    #     >>> col2 = ['3','4000']
    #     >>> print(_table([col1,col2]))
    #     ┌─┬────────┐
    #     │1│3       │
    #     │ │        │
    #     ├─┼────────┤
    #     │2│40000000│
    #     └─┴────────┘

    cols = rows if transpose else [list(c) for c in zip(*rows)]

    for i, _cells in enumerate(zip(*cols)):
        max_cell_height = max(map(lambda x: x.count("\n"), _cells))
        for j in range(len(_cells)):
            cols[j][i] += "\n" * (max_cell_height - cols[j][i].count("\n"))

    return _hstack(*(_vbox(*col) for col in cols))


def size_pp(size: int, **spec: Unpack[PPSpec]):
    del spec
    order_alpha = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    size_order = int(math.log(size, 1024)) if size else 0
    text = f"{(size)/(1024**size_order):.2f}{order_alpha[size_order]}"
    return text


# `tree_summary`` display dispatchers to make `tree_summary` more customizable
# for type, count, and size display.
# for example, array count uses the array size instead, where as some other
# custom object might make use of this feature to display the number of elements
# differently. for size, only arrays can have a size, as other type sizes are
# not meaningful. user can define their own dispatchers for their custom types
# display e.g `f32[10,10]` instead of simply `Array`.


@ft.singledispatch
def type_dispatcher(node: Any) -> str:
    return type(node).__name__


@type_dispatcher.register(np.ndarray)
@type_dispatcher.register(jax.Array)
@type_dispatcher.register(jax.ShapeDtypeStruct)
def _(node: Any) -> str:
    """Return the type repr of the node."""
    shape_dype = node.shape, node.dtype
    spec = dict(indent=0, kind="REPR", width=80, depth=float("inf"), seen=set())
    return pp(jax.ShapeDtypeStruct(*shape_dype), **spec)


@ft.singledispatch
def count_dispatcher(_: Any) -> int:
    """Return the number of elements in a node."""
    return 1


@count_dispatcher.register(jax.Array)
@count_dispatcher.register(np.ndarray)
def _(node: jax.Array | np.ndarray) -> int:
    return node.size


@ft.singledispatch
def size_dispatcher(node: Any) -> None:
    """Return the size of a node in bytes."""
    return 0


@size_dispatcher.register(jax.Array)
@size_dispatcher.register(np.ndarray)
def _(node: jax.Array | np.ndarray) -> int:
    return node.nbytes


def tree_size(tree: PyTree) -> int:
    def reduce_func(acc, node):
        return acc + size_dispatcher(node)

    return jtu.tree_reduce(reduce_func, tree, initializer=0)


def tree_count(tree: PyTree) -> int:
    def reduce_func(acc, node):
        return acc + count_dispatcher(node)

    return jtu.tree_reduce(reduce_func, tree, initializer=0)


def tree_summary(
    tree: PyTree,
    *,
    depth: int | float = float("inf"),
    is_leaf: IsLeafType = None,
) -> str:
    """Print a summary of an arbitrary PyTree.

    Args:
        tree: a jax registered pytree to summarize.
        depth: max depth to display the tree. defaults to maximum depth.
        is_leaf: function to determine if a node is a leaf. defaults to None

    Returns:
        String summary of the tree structure
        - First column: path to the node.
        - Second column: type of the node. to control the displayed type use
            `tree_summary.def_type(type, func) to define a custom type display function.
        - Third column: number of leaves in the node. for arrays the number of leaves
            is the number of elements in the array, otherwise its 1. to control the
            number of leaves of a node use `tree_summary.def_count(type,func)`
        - Fourth column: size of the node in bytes. if the node is array the size
            is the size of the array in bytes, otherwise its the size is not displayed.
            to control the size of a node use `tree_summary.def_size(type,func)`
        - Last row: type of parent, number of leaves of the parent

    Example:
        >>> import pytreeclass as pytc
        >>> import jax.numpy as jnp
        >>> print(pytc.tree_summary([1, [2, [3]], jnp.array([1, 2, 3])]))
        ┌─────────┬──────┬─────┬──────┐
        │Name     │Type  │Count│Size  │
        ├─────────┼──────┼─────┼──────┤
        │[0]      │int   │1    │      │
        ├─────────┼──────┼─────┼──────┤
        │[1][0]   │int   │1    │      │
        ├─────────┼──────┼─────┼──────┤
        │[1][1][0]│int   │1    │      │
        ├─────────┼──────┼─────┼──────┤
        │[2]      │i32[3]│3    │12.00B│
        ├─────────┼──────┼─────┼──────┤
        │Σ        │list  │6    │12.00B│
        └─────────┴──────┴─────┴──────┘

    Example:
        >>> # set python `int` to have 4 bytes using dispatching
        >>> import pytreeclass as pytc
        >>> print(pytc.tree_summary(1))
        ┌────┬────┬─────┬────┐
        │Name│Type│Count│Size│
        ├────┼────┼─────┼────┤
        │Σ   │int │1    │    │
        └────┴────┴─────┴────┘
        >>> @pytc.tree_summary.def_size(int)
        ... def _(node: int) -> int:
        ...     return 4
        >>> print(pytc.tree_summary(1))
        ┌────┬────┬─────┬─────┐
        │Name│Type│Count│Size │
        ├────┼────┼─────┼─────┤
        │Σ   │int │1    │4.00B│
        └────┴────┴─────┴─────┘
    """
    rows = [["Name", "Type", "Count", "Size"]]
    tcount = tsize = 0

    # use `unzip2` from `jax.util` to avoid [] leaves
    # based on this issue:
    traces, leaves = unzip2(
        tree_leaves_with_trace(
            tree,
            is_leaf=is_leaf,
            is_trace_leaf=_is_trace_leaf_depth_factory(depth),
        )
    )

    for trace, leaf in zip(traces, leaves):
        tcount += (count := tree_count(leaf))
        tsize += (size := tree_size(leaf))

        if trace == ((), ()):
            # avoid printing the leaf trace (which is the root of the tree)
            # twice, once as a leaf and once as the root at the end
            continue

        paths, _ = trace
        pstr = jtu.keystr(paths)
        tstr = type_dispatcher(leaf)
        cstr = f"{count:,}" if count else ""
        sstr = size_pp(size) if size else ""
        rows += [[pstr, tstr, cstr, sstr]]

    pstr = "Σ"
    tstr = type_dispatcher(tree)
    cstr = f"{tcount:,}" if tcount else ""
    sstr = size_pp(tsize) if tsize else ""
    rows += [[pstr, tstr, cstr, sstr]]
    return _table(rows)


tree_summary.def_count = count_dispatcher.register
tree_summary.def_size = size_dispatcher.register
tree_summary.def_type = type_dispatcher.register


def tree_repr_with_trace(
    tree: PyTree,
    is_leaf: IsLeafType = None,
    transpose: bool = False,
) -> PyTree:
    """Return a pytree with leaf nodes replaced with their trace.

    Args:
        tree: pytree to summarize.
        is_leaf: function to determine if a node is a leaf. defaults to None
        transpose: transpose the table. i.e. rows become cols and cols become rows

    Example:
        >>> import pytreeclass as pytc
        >>> class Test(pytc.TreeClass):
        ...    a:int = 1
        ...    b:float = 2.0

        >>> tree = Test()
        >>> print(pytc.tree_repr_with_trace(Test()))  # doctest: +NORMALIZE_WHITESPACE
        Test(
          a=
            ┌─────────┬───┐
            │Value    │1  │
            ├─────────┼───┤
            │Name path│.a │
            ├─────────┼───┤
            │Type path│int│
            └─────────┴───┘,
          b=
            ┌─────────┬─────┐
            │Value    │2.0  │
            ├─────────┼─────┤
            │Name path│.b   │
            ├─────────┼─────┤
            │Type path│float│
            └─────────┴─────┘
        )

        >>> print(pytc.tree_repr_with_trace(Test(), transpose=True)) # doctest: +NORMALIZE_WHITESPACE
        Test(
          a=
            ┌─────┬─────────┬─────────┐
            │Value│Name path│Type path│
            ├─────┼─────────┼─────────┤
            │1    │.a       │int      │
            └─────┴─────────┴─────────┘,
          b=
            ┌─────┬─────────┬─────────┐
            │Value│Name path│Type path│
            ├─────┼─────────┼─────────┤
            │2.0  │.b       │float    │
            └─────┴─────────┴─────────┘
        )

    Note:
        This function can be useful for debugging and raising descriptive errors.
    """

    def leaf_trace_summary(trace, leaf) -> str:
        # this can be useful in debugging and raising descriptive errors

        rows = [["Value", tree_repr(leaf)]]

        names = "->".join(str(i) for i in trace[0])
        rows += [["Name path", names]]

        types = "->".join(i.__name__ for i in trace[1])
        rows += [["Type path", types]]

        # make a pretty table for each leaf
        return "\n\t" + ("\n\t").join(_table(rows, transpose=transpose).split("\n"))

    return tree_map_with_trace(leaf_trace_summary, tree, is_leaf=is_leaf)
