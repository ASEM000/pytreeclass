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
from contextlib import suppress
from itertools import zip_longest
from types import FunctionType
from typing import Any, Callable, Iterable, Literal, Sequence

import jax
import jax.tree_util as jtu
import numpy as np
from jax import custom_jvp
from jax.util import unzip2
from typing_extensions import TypeAlias, TypedDict, Unpack

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

    with suppress(Exception):
        # maybe the array is a jax tracers
        low, high = np.min(node), np.max(node)

        interval = (
            ("(" if math.isinf(low) else "[")
            + (
                f"{low},{high}"
                if issubclass(node.dtype.type, np.integer)
                else f"{low:.2f},{high:.2f}"
            )
            + (")" if math.isinf(high) else "]")
        ).replace("inf", "∞")

        mean, std = f"{np.mean(node):.2f}", f"{np.std(node):.2f}"
        return f"{base}(μ={mean}, σ={std}, ∈{interval})"

    return base


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


@pp_dispatcher.register(str)
def str_pp(node: str, **spec: Unpack[PPSpec]) -> str:
    return node


def tree_repr(
    tree: PyTree,
    *,
    width: int = 80,
    tabwidth: int = 2,
    depth: int | float = float("inf"),
) -> str:
    """Prertty print arbitrary pytrees `__repr__`.

    Args:
        tree: arbitrary pytree.
        width: max width of the repr string.
        tabwidth: tab width of the repr string.
        depth: max depth of the repr string.

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
    """Prertty print arbitrary pytrees `__str__`.

    Args:
        tree: arbitrary pytree.
        width: max width of the str string.
        tabwidth: tab width of the repr string.
        depth: max depth of the repr string.

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


def tree_diagram(
    tree: Any,
    *,
    depth: int | float = float("inf"),
    is_leaf: IsLeafType = None,
    tabwidth: int = 4,
):
    """Pretty print arbitrary pytrees tree with tree structure diagram.

    Args:
        tree: arbitrary pytree.
        depth: depth of the tree to print. default is max depth.
        is_leaf: function to determine if a node is a leaf. default is None.
        tabwidth: tab width of the repr string. default is 4.

    Example:
        >>> import pytreeclass as pytc
        >>> @pytc.autoinit
        ... class A(pytc.TreeClass):
        ...        x: int = 10
        ...        y: int = (20,30)
        ...        z: int = 40

        >>> @pytc.autoinit
        ... class B(pytc.TreeClass):
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
            (key, _), value = node.data
            text = f"{indent}"
            text += f"{branch}{key}=" if key is not None else ""
            text += tree_repr(value, depth=0)
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
    """Generate a mermaid diagram syntax for arbitrary pytrees.

    Args:
        tree: PyTree
        depth: depth of the tree to print. default is max depth
        is_leaf: function to determine if a node is a leaf. default is None
        tabwidth: tab width of the repr string. default is 4.

    Note:
        - Copy the output and paste it in the mermaid live editor to interact with
          the diagram. https://mermaid.live
    """

    def step(node: Node, depth: int = 0) -> str:
        if len(node.children) == 0:
            (key, _), value = node.data
            ppstr = f"{key}=" if key is not None else ""
            ppstr += tree_repr(value, depth=0)
            ppstr = "<b>" + ppstr + "</b>"
            return f'\tid{id(node.parent)} --- id{id(node)}("{ppstr}")\n'

        (key, type), _ = node.data
        ppstr = f"{key}:" if key is not None else ""
        ppstr += f"{type.__name__}"
        ppstr = "<b>" + ppstr + "</b>"

        if node.parent is None:
            text = f'\tid{id(node)}("{ppstr}")\n'
        else:
            text = f'\tid{id(node.parent)} --- id{id(node)}("{ppstr}")\n'

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


# dispatcher for dot nodestyles
dot_dispatcher = ft.singledispatch(lambda _: dict(shape="box"))


def tree_graph(
    tree: PyTree,
    depth: int | float = float("inf"),
    is_leaf: IsLeafType = None,
    tabwidth: int | None = 4,
) -> str:
    """Generate a dot diagram syntax for arbitrary pytrees.

    Args:
        tree: pytree
        depth: depth of the tree to print. default is max depth
        is_leaf: function to determine if a node is a leaf. default is None
        tabwidth: tab width of the repr string. default is 4.

    Returns:
        str: dot diagram syntax

    Example:
        >>> import pytreeclass as pytc
        >>> tree = [1, 2, dict(a=3)]
        >>> print(pytc.tree_graph(tree))  # doctest: +SKIP

        .. graphviz::

            digraph G {
                4685268864 [label="list", shape=box];
                4685269056 [label="[0]=1", shape=box];
                4685268864 -> 4685269056;
                4685269120 [label="[1]=2", shape=box];
                4685268864 -> 4685269120;
                4685269184 [label="[2]:dict", shape=box];
                4685268864 -> 4685269184;
                4685269248 [label="['a']=3", shape=box];
                4685269184 -> 4685269248;
            }

    Example:
        >>> # define custom style for a node by dispatching on the value
        >>> # the defined function should return a dict of attributes
        >>> # that will be passed to graphviz.
        >>> import pytreeclass as pytc
        >>> tree = [1, 2, dict(a=3)]
        >>> @pytc.tree_graph.def_nodestyle(list)
        ... def _(_) -> dict[str, str]:
        ...     return dict(shape="circle", style="filled", fillcolor="lightblue")
        >>> print(pytc.tree_graph(tree))  # doctest: +SKIP

        .. graphviz::

            digraph G {
                4685309312 [label="list", shape=circle, style=filled, fillcolor=lightblue];
                4685309504 [label="[0]=1", shape=box];
                4685309312 -> 4685309504;
                4685309568 [label="[1]=2", shape=box];
                4685309312 -> 4685309568;
                4685309632 [label="[2]:dict", shape=box];
                4685309312 -> 4685309632;
                4685309696 [label="['a']=3", shape=box];
                4685309632 -> 4685309696;
            }
    """

    def step(node: Node, depth: int = 0) -> str:
        (key, type), value = node.data

        # dispatch node style
        style = ", ".join(f"{k}={v}" for k, v in dot_dispatcher(value).items())

        if len(node.children) == 0:
            ppstr = f"{key}=" if key is not None else ""
            ppstr += tree_repr(value, depth=0)
            text = f'\t{id(node)} [label="{ppstr}", {style}];\n'
            text += f"\t{id(node.parent)} -> {id(node)};\n"
            return text

        ppstr = f"{key}:" if key is not None else ""
        ppstr += f"{type.__name__}"

        if node.parent is None:
            text = f'\t{id(node)} [label="{ppstr}", {style}];\n'
        else:
            text = f'\t{id(node)} [label="{ppstr}", {style}];\n'
            text += f"\t{id(node.parent)} -> {id(node)};\n"

        for child in node.children.values():
            text += step(child, depth=depth + 1)
        return text

    root = construct_tree(
        tree,
        is_leaf=is_leaf,
        is_trace_leaf=_is_trace_leaf_depth_factory(depth),
    )
    text = "digraph G {\n" + step(root) + "}"
    return (text.expandtabs(tabwidth) if tabwidth is not None else text).rstrip()


tree_graph.def_nodestyle = dot_dispatcher.register


def format_width(string, width=60):
    """Strip newline/tab characters if less than max width."""
    children_length = len(string) - string.count("\n") - string.count("\t")
    if children_length > width:
        return string
    return string.replace("\n", "").replace("\t", "")


# table printing

Row: TypeAlias = Sequence[str]  # list of columns


def _table(rows: list[Row]) -> str:
    """Generate a table from a list of rows."""

    def line(text: Row, widths: list[int]) -> str:
        return "\n".join(
            "│"
            + "│".join(col.ljust(width) for col, width in zip(line_row, widths))
            + "│"
            for line_row in zip_longest(*[t.split("\n") for t in text], fillvalue="")
        )

    widths = [max(map(len, "\n".join(col).split("\n"))) for col in zip(*rows)]
    spaces: Row = ["─" * width for width in widths]

    return (
        ("┌" + "┬".join(spaces) + "┐")
        + "\n"
        + ("\n├" + "┼".join(spaces) + "┤\n").join(line(row, widths) for row in rows)
        + "\n"
        + ("└" + "┴".join(spaces) + "┘")
    )


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
        String summary of the tree structure:
            - First column: path to the node.
            - Second column: type of the node. to control the displayed type use
                `tree_summary.def_type(type, func)` to define a custom type display function.
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

    Example:
        >>> # set custom type display for jaxprs
        >>> import jax
        >>> import pytreeclass as pytc
        >>> ClosedJaxprType = type(jax.make_jaxpr(lambda x: x)(1))
        >>> @pytc.tree_summary.def_type(ClosedJaxprType)
        ... def _(expr: ClosedJaxprType) -> str:
        ...     jaxpr = expr.jaxpr
        ...     return f"Jaxpr({jaxpr.invars}, {jaxpr.outvars})"
        >>> def func(x, y):
        ...     return x
        >>> jaxpr = jax.make_jaxpr(func)(1, 2)
        >>> print(pytc.tree_summary(jaxpr))
        ┌────┬──────────────────┬─────┬────┐
        │Name│Type              │Count│Size│
        ├────┼──────────────────┼─────┼────┤
        │Σ   │Jaxpr([a, b], [a])│1    │    │
        └────┴──────────────────┴─────┴────┘
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
        tstr = tree_summary.type_dispatcher(leaf)
        cstr = f"{count:,}" if count else ""
        sstr = size_pp(size) if size else ""
        rows += [[pstr, tstr, cstr, sstr]]

    pstr = "Σ"
    tstr = tree_summary.type_dispatcher(tree)
    cstr = f"{tcount:,}" if tcount else ""
    sstr = size_pp(tsize) if tsize else ""
    rows += [[pstr, tstr, cstr, sstr]]
    return _table(rows)


tree_summary.count_dispatcher = ft.singledispatch(lambda x: 1)
tree_summary.def_count = tree_summary.count_dispatcher.register
tree_summary.size_dispatcher = ft.singledispatch(lambda x: 0)
tree_summary.def_size = tree_summary.size_dispatcher.register
tree_summary.type_dispatcher = ft.singledispatch(lambda x: type(x).__name__)
tree_summary.def_type = tree_summary.type_dispatcher.register


@tree_summary.def_type(np.ndarray)
@tree_summary.def_type(jax.Array)
@tree_summary.def_type(jax.ShapeDtypeStruct)
def _(node: Any) -> str:
    """Return the type repr of the node."""
    shape_dype = node.shape, node.dtype
    spec = dict(indent=0, kind="REPR", width=80, depth=float("inf"), seen=set())
    return pp(jax.ShapeDtypeStruct(*shape_dype), **spec)


@tree_summary.def_count(jax.Array)
@tree_summary.def_count(np.ndarray)
def _(node: jax.Array | np.ndarray) -> int:
    return node.size


@tree_summary.def_size(jax.Array)
@tree_summary.def_size(np.ndarray)
def _(node: jax.Array | np.ndarray) -> int:
    return node.nbytes


def tree_size(tree: PyTree) -> int:
    def reduce_func(acc, node):
        return acc + tree_summary.size_dispatcher(node)

    return jtu.tree_reduce(reduce_func, tree, initializer=0)


def tree_count(tree: PyTree) -> int:
    def reduce_func(acc, node):
        return acc + tree_summary.count_dispatcher(node)

    return jtu.tree_reduce(reduce_func, tree, initializer=0)


def tree_repr_with_trace(
    tree: PyTree,
    is_leaf: IsLeafType = None,
    transpose: bool = False,
) -> PyTree:
    """Return a pytree with leaf nodes replaced with their trace.

    Args:
        tree: pytree to summarize.
        is_leaf: function to determine if a node is a leaf. defaults to None.
        transpose: transpose the table. i.e. rows become cols and cols become rows.

    Example:
        >>> import pytreeclass as pytc
        >>> @pytc.autoinit
        ... class Test(pytc.TreeClass):
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

        joiner = "\n" + "\t" * (len(trace[0]) + 1)

        # make a pretty table for each leaf
        rows = list(map(list, zip(*rows))) if transpose else rows
        return joiner + (joiner).join(_table(rows).split("\n"))

    return tree_map_with_trace(leaf_trace_summary, tree, is_leaf=is_leaf)
