from __future__ import annotations

import ctypes
import dataclasses as dc
import functools as ft
import inspect
import math
from types import FunctionType
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax._src.custom_derivatives import custom_jvp
from jaxlib.xla_extension import CompiledFunction

import pytreeclass as pytc
from pytreeclass._src.tree_viz_util import (
    NodeInfo,
    _calculate_node_info_stats,
    _format_count,
    _format_size,
    _format_width,
    _marker,
    _mermaid_marker,
    _table,
    _tree_trace,
)

PyTree = Any


def _node_pprint(node: Any, depth: int = 0, kind: str = "repr") -> str:
    if isinstance(node, ft.partial):
        # applies for partial functions including `jax.tree_util.Partial`
        return f"Partial({_func_pprint(node.func)})"

    if isinstance(node, (FunctionType, custom_jvp)):
        return _func_pprint(node)

    if isinstance(node, CompiledFunction):
        # special case for jitted JAX functions
        return f"jit({_func_pprint(node)})"

    if isinstance(node, (np.ndarray, jnp.ndarray)):
        # works for numpy arrays, jax arrays
        return _numpy_pprint(node, depth, kind)

    if isinstance(node, jax.ShapeDtypeStruct):
        return _shape_dtype_struct_pprint(node)

    if hasattr(node, "_fields") and hasattr(node, "_asdict"):
        return _namedtuple_pprint(node, depth, kind=kind)

    if isinstance(node, list):
        return _list_pprint(node, depth, kind=kind)

    if isinstance(node, tuple):
        return _tuple_pprint(node, depth, kind=kind)

    if isinstance(node, set):
        return _set_pprint(node, depth, kind=kind)

    if isinstance(node, dict):
        return _dict_pprint(node, depth, kind=kind)

    if dc.is_dataclass(node):
        return _marked_dataclass_pprint(node, depth, kind=kind)

    if isinstance(node, slice):
        return _slice_pprint(node)

    return _general_pprint(node, depth, kind=kind)


_printer_map = {
    "repr": lambda node, depth: _node_pprint(node, depth, kind="repr"),
    "str": lambda node, depth: _node_pprint(node, depth, kind="str"),
}


def _general_pprint(node: Any, depth: int, kind: str) -> str:
    if isinstance(node, object) and node.__class__.__repr__ is not object.__repr__:
        # use custom __repr__ method if available
        fmt = f"{node!r}" if kind == "repr" else f"{node!s}"
    elif isinstance(node, type):
        fmt = f"{node!r}" if kind == "repr" else f"{node!s}"
    else:
        # use `jax.tree_util.tree_map`, to get representation of the node
        printer = _printer_map[kind]
        leaves = jtu.tree_leaves(node)
        fmt = ", ".join(f"leaf_{i}={printer(v,depth)}" for i, v in enumerate(leaves))
        fmt = f"{node.__class__.__name__}({fmt})"

    is_mutltiline = "\n" in fmt

    # multiline repr/str case, increase depth and indent
    depth = (depth + 1) if is_mutltiline else depth
    fmt = ("\n" + "\t" * depth).join(fmt.split("\n"))

    return ("\n" + "\t" * (depth) + fmt) if is_mutltiline else fmt


def _shape_dtype_struct_pprint(node: jax.ShapeDtypeStruct) -> str:
    """Pretty print jax.ShapeDtypeStruct"""
    shape = (
        f"{node.shape}".replace(",", "")
        .replace("(", "[")
        .replace(")", "]")
        .replace(" ", ",")
    )
    if issubclass(node.dtype.type, np.integer):
        return f"{node.dtype}".replace("int", "i") + shape
    elif issubclass(node.dtype.type, np.floating):
        return f"{node.dtype}".replace("float", "f") + shape
    elif issubclass(node.dtype.type, np.complexfloating):
        return f"{node.dtype}".replace("complex", "c") + shape
    return f"{node.dtype}" + shape


def _numpy_pprint(node: np.ndarray, depth: int, kind: str = "repr") -> str:
    """Replace np.ndarray repr with short hand notation for type and shape

    Example:
        >>> _numpy_pprint(np.ones((2,3)))
        'f64[2,3]∈[1.0,1.0]'
    """
    if kind == "str":
        return _general_pprint(node, depth, kind="str")

    base = _shape_dtype_struct_pprint(node)

    # Extended repr for numpy array, with extended information
    # this part of the function is inspired by
    # lovely-jax https://github.com/xl0/lovely-jax

    if issubclass(node.dtype.type, np.number):
        # get min, max, mean, std of node
        low, high = np.min(node), np.max(node)
        # add brackets to indicate closed/open interval
        interval = "(" if math.isinf(low) else "["
        # if issubclass(node.dtype.type, np.integer):
        # if integer, round to nearest integer
        interval += (
            f"{low},{high}"
            if issubclass(node.dtype.type, np.integer)
            else f"{low:.1f},{high:.1f}"
        )

        mean, std = np.mean(node), np.std(node)
        # add brackets to indicate closed/open interval
        interval += ")" if math.isinf(high) else "]"
        # replace inf with infinity symbol
        interval = interval.replace("inf", "∞")
        # return extended repr
        return f"{base} ∈{interval} μ(σ)={mean:.1f}({std:.1f})"

    # kind is repr
    return base


@ft.lru_cache
def _func_pprint(func: Callable) -> str:
    """Pretty print function

    Example:
        >>> def example(a: int, b=1, *c, d, e=2, **f) -> str:
            ...
        >>> _func_pprint(example)
        "example(a, b, *c, d, e, **f)"
    """
    args, varargs, varkw, _, kwonlyargs, _, _ = inspect.getfullargspec(func)
    args = (", ".join(args)) if len(args) > 0 else ""
    varargs = ("*" + varargs) if varargs is not None else ""
    kwonlyargs = (", ".join(kwonlyargs)) if len(kwonlyargs) > 0 else ""
    varkw = ("**" + varkw) if varkw is not None else ""
    name = "Lambda" if (func.__name__ == "<lambda>") else func.__name__

    fmt = f"{name}("
    fmt += ", ".join(item for item in [args, varargs, kwonlyargs, varkw] if item != "")
    fmt += ")"
    return fmt


def _slice_pprint(node: slice) -> str:
    start = node.start if node.start is not None else ""
    stop = node.stop if node.stop is not None else ""
    step = node.step if node.step is not None else ""

    if step and step > 1:
        return f"[{start}:{stop}:{step}]"

    if stop == start + 1:
        return f"[{start}]"

    return f"[{start}:{stop}]"


def _list_pprint(node: list, depth: int, kind: str = "repr") -> str:
    """Pretty print a list"""
    printer = _printer_map[kind]
    fmt = (f"{_format_width(printer(v,depth=depth+1))}" for v in node)
    fmt = (", \n" + "\t" * (depth + 1)).join(fmt)
    fmt = "[\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + "]"
    return _format_width(fmt)


def _tuple_pprint(node: tuple, depth: int, kind: str = "repr") -> str:
    """Pretty print a list"""
    printer = _printer_map[kind]
    fmt = (f"{_format_width(printer(v,depth=depth+1))}" for v in node)
    fmt = (", \n" + "\t" * (depth + 1)).join(fmt)
    fmt = "(\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + ")"
    return _format_width(fmt)


def _set_pprint(node: set, depth: int, kind: str = "repr") -> str:
    """Pretty print a list"""
    printer = _printer_map[kind]
    fmt = (f"{_format_width(printer(v,depth=depth+1))}" for v in node)
    fmt = (", \n" + "\t" * (depth + 1)).join(fmt)
    fmt = "{\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + "}"
    return _format_width(fmt)


def _dict_pprint(node: dict, depth: int, kind: str = "repr") -> str:
    printer = _printer_map[kind]
    fmt = (f"{k}:{printer(v,depth=depth+1)}" for k, v in node.items())
    fmt = (", \n" + "\t" * (depth + 1)).join(fmt)
    fmt = "{\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + "}"
    return _format_width(fmt)


def _namedtuple_pprint(node, depth: int, kind: str = "repr") -> str:
    printer = _printer_map[kind]
    fmt = (f"{k}={printer(v,depth=depth+1)}" for k, v in node._asdict().items())
    fmt = (", \n" + "\t" * (depth + 1)).join(fmt)
    fmt = "namedtuple(\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + ")"
    return _format_width(fmt)


@ft.lru_cache
def _marked_dataclass_pprint(node: dict, depth: int, kind: str = "repr") -> str:
    printer = _printer_map[kind]
    name = node.__class__.__name__
    vs = (getattr(node, f.name) for f in dc.fields(node) if f.repr)
    fs = (f for f in dc.fields(node) if f.repr)
    fmt = (f"{f.name}={_marker(f,v)}{printer(v,depth+1)}" for f, v in zip(fs, vs))
    fmt = (", \n" + "\t" * (depth + 1)).join(fmt)
    fmt = f"{name}(\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + ")"
    return _format_width(fmt)


def _node_type_pprint(node):
    if hasattr(node, "dtype") and hasattr(node, "shape"):
        return _node_pprint(jax.ShapeDtypeStruct(node.shape, node.dtype))
    return f"{node.__class__.__name__}"


def tree_repr(tree, width: int = 80) -> str:
    """Prertty print dataclass tree __repr__"""
    return _node_pprint(tree, depth=0, kind="repr").expandtabs(2)


def tree_str(tree, width: int = 80) -> str:
    """Prertty print dataclass tree __str__"""
    return _node_pprint(tree, depth=0, kind="str").expandtabs(2)


class _TreePretty:
    def __repr__(self) -> str:
        return tree_repr(self)

    def __str__(self) -> str:
        return tree_str(self)


def tree_diagram(tree, depth=float("inf")):
    """Pretty print treeclass tree with tree structure diagram

    Args:
        tree: PyTree
        depth: depth of the tree to print. default is max depth

    Example:
        >>> @pytc.treeclass
        ... class A:
        ...        x: int = 10
        ...        y: int = (20,30)
        ...        z: int = 40

        >>> @pytc.treeclass
        ... class B:
        ...     a: int = 10
        ...     b: tuple = (20,30, A())

        >>> print(tree_diagram(B()), depth=0)
        B

        >>> print(pytc.tree_diagram(B(), depth=1))
        B
            ├── a:int=10
            └── b:tuple=(20, 30, A(x=10, y=(20, 30), z=40))


        >>> print(pytc.tree_diagram(B(), depth=2))
        B
            ├── a:int=10
            └── b:tuple
                ├── [0]:int=20
                ├── [1]:int=30
                └── [2]:A=A(x=10, y=(20, 30), z=40)

        >>> print(pytc.tree_diagram(B(), depth=3))
        B
            ├── a:int=10
            └── b:tuple
                ├── [0]:int=20
                ├── [1]:int=30
                └── [2]:A
                    ├── x:int=10
                    ├── y:tuple=(20, 30)
                    └── z:int=40

        >>> print(pytc.tree_diagram(B(), depth=4))
        B
            ├── a:int=10
            └── b:tuple
                ├── [0]:int=20
                ├── [1]:int=30
                └── [2]:A
                    ├── x:int=10
                    ├── y:tuple
                    │   ├── [0]:int=20
                    │   └── [1]:int=30
                    └── z:int=40
    """

    infos = _tree_trace(tree, depth)
    fmt = f"{tree.__class__.__name__}"
    printer = _printer_map["repr"]

    for i, info in enumerate(infos):
        # iterate over the leaves `NodeInfo` object

        max_depth = len(info.names)
        index = info.index

        for depth, (name, type) in enumerate(zip(info.names, info.types)):
            # skip printing the common parent node twice
            if i > 0 and infos[i - 1].names[: depth + 1] == info.names[: depth + 1]:
                continue

            fmt += "\n\t"
            fmt += "".join(("" if index[i] == 0 else "│") + "\t" for i in range(depth))
            fmt += "├" if not index[depth] == 0 else "└"
            mark = "#" if (info.frozen and depth == max_depth - 1) else "─"
            fmt += f"{mark}─ {printer(name,depth)}:{type.__name__}"
            fmt += f"={printer(info.node,depth+2)}" if depth == max_depth - 1 else ""

    return fmt.expandtabs(4)


def tree_summary(tree: PyTree, *, depth=float("inf")) -> str:
    """Print a summary of a pytree structure

    Args:
        tree: pytree to summarize (ex. list, tuple, dict, dataclass, jax.numpy.ndarray)
        depth: depth to traverse the tree. defaults to maximum depth.

    Returns:
        str: summary of the tree structure
            1st column: is the path to the node
            2nd column: is the type of the node
            3rd column: is the number of leaves in the node (the number of frozen leaves displayed between parenthesis)
            4th column: is the size of the node (the size of frozen leaves displayed between parenthesis)
            Last row  : type of parent, number of leaves and size of parent

    Note:
        Array elements are considered as leaves, for example `jnp.array([1,2,3])` has 3 leaves

    Example:
        >>> print(pytc.tree_summary([1,[2,[3]]]))
        ┌─────────┬────┬─────────────┬─────────────┐
        │Name     │Type│Count(Frozen)│Size(Frozen) │
        ├─────────┼────┼─────────────┼─────────────┤
        │[0]      │int │1(0)         │28.00B(0.00B)│
        ├─────────┼────┼─────────────┼─────────────┤
        │[1][0]   │int │1(0)         │28.00B(0.00B)│
        ├─────────┼────┼─────────────┼─────────────┤
        │[1][1][0]│int │1(0)         │28.00B(0.00B)│
        ├─────────┼────┼─────────────┼─────────────┤
        │Σ        │list│3(0)         │84.00B(0.00B)│
        └─────────┴────┴─────────────┴─────────────┘
    """

    ROWS = [["Name", "Type", "Count(Frozen)", "Size(Frozen)"]]

    # traverse the tree and collect info about each node
    infos = _tree_trace(tree, depth)
    # in case of a single node tree or depth=0, avoid printing the node twice
    # once for the trace and once for the summary
    infos = infos if len(infos) > 1 else ()

    for info in infos:
        if not info.repr:
            # skip nodes that are not repr
            continue

        path = ".".join(_node_pprint(i, kind="str") for i in info.names)
        row = [path.replace("].", "]").replace(".[", "[")]

        # type name row
        row += [_node_type_pprint(pytc.unfreeze(info.node))]

        # count and size row
        count, size = _calculate_node_info_stats(info)
        row += [f"{_format_count(count.real+count.imag)}({_format_count(count.imag)})"]
        row += [f"{_format_size(size.real+size.imag)}({_format_size(size.imag)})"]

        ROWS += [row]

    # add summary row at the end to
    # show total number of leaves and total size
    COUNT = [complex(0), complex(0)]  # non-frozen, frozen
    SIZE = [complex(0), complex(0)]

    for info in _tree_trace(tree):
        count, size = _calculate_node_info_stats(info)
        COUNT[info.frozen] += count
        SIZE[info.frozen] += size

    unfrozen_count = COUNT[0].real + COUNT[0].imag
    frozen_count = COUNT[1].real + COUNT[1].imag
    total_count = unfrozen_count + frozen_count

    unfrozen_size = SIZE[0].real + SIZE[0].imag
    frozen_size = SIZE[1].real + SIZE[1].imag
    total_size = unfrozen_size + frozen_size

    row = ["Σ"]
    row += [_node_type_pprint(tree)]
    row += [f"{_format_count(total_count)}({_format_count(frozen_count)})"]
    row += [f"{_format_size(total_size)}({_format_size(frozen_size)})"]
    ROWS += [row]

    COLS = [list(c) for c in zip(*ROWS)]
    layer_table = _table(COLS)
    return layer_table.expandtabs(8)


def tree_mermaid(tree: PyTree):
    # def _generate_mermaid_link(mermaid_string: str) -> str:
    #     """generate a one-time link mermaid diagram"""
    #     url_val = "https://pytreeclass.herokuapp.com/generateTemp"
    #     request = requests.post(url_val, json={"description": mermaid_string})
    #     generated_id = request.json()["id"]
    #     generated_html = f"https://pytreeclass.herokuapp.com/temp/?id={generated_id}"
    #     return f"Open URL in browser: {generated_html}"

    def bold_text(text: str) -> str:
        # bold a text in ansci code
        return "<b>" + text + "</b>"

    def node_id(input):
        """hash a value by its location in a tree. used to connect values in mermaid"""
        return ctypes.c_size_t(hash(input)).value

    def reduce_count_and_size(item: Any) -> tuple[complex, complex]:
        def reduce_func(acc, cur: NodeInfo) -> tuple[complex, complex]:
            count, size = acc
            return (count + cur.count, size + cur.size)

        return ft.reduce(reduce_func, _tree_trace(item), (0, 0))

    def recurse(tree, depth, prev_id, expand=True):
        nonlocal fmt

        if dc.is_dataclass(tree) and expand:
            fields = [f for f in dc.fields(tree) if f.repr]
            names = (f.name for f in fields)
            values = (getattr(tree, f.name) for f in fields)
            marks = (_mermaid_marker(f, getattr(tree, f.name), default="--->") for f in fields)  # fmt: skip

        elif isinstance(tree, (list, tuple)) and expand:
            # expand lists and tuples
            names = (f"[{i}]" for i in range(len(tree)))
            values = tree
            marks = ("--->" for _ in tree)

        elif isinstance(tree, dict) and expand:
            names = tree.keys()
            values = tree.values()
            marks = ("--->" for _ in tree)
        else:
            return

        for i, (value, name, mark) in enumerate(zip(values, names, marks)):
            cur_id = node_id((depth, -1, prev_id))
            count, size = reduce_count_and_size(value)
            count = count.real + count.imag
            size = size.real + size.imag

            if count == 0:
                count = size = ""
            else:
                count = _format_count(count) + (" leaves" if count > 1 else " leaf")
                size = _format_size(size.real)

            cur_id = node_id((depth, i, prev_id))
            type = value.__class__.__name__

            if dc.is_dataclass(value) or (
                isinstance(value, (list, tuple)) and any(map(dc.is_dataclass, value))
            ):

                fmt += f"\n\tid{prev_id} {mark} "
                fmt += f'|"{(count)}<br>{(size)}"| '  # add count and size of children
                fmt += f'id{cur_id}("{bold_text(name) }:{type}")'
                recurse(tree=value, depth=depth + 1, prev_id=cur_id, expand=True)
                continue

            fmt += f"\n\tid{prev_id} {mark} "

            # add count and size of children
            if mark == "-..-":
                fmt += f'|"(frozen) {(count)}<br>{(size)}"| '  # add count and size of children for frozen leaves
            elif mark == "--x":
                fmt += ""  # no count or size for frozen leaves
            elif mark == "--->" or mark == "---":
                fmt += f'|"{(count)}<br>{(size)}"| '

            fmt += f'id{cur_id}["{bold_text(name)}:{type}={_node_pprint(value, kind="repr")}"]'

        prev_id = cur_id

    cur_id = node_id((0, 0, -1, 0))
    fmt = f"flowchart LR\n\tid{cur_id}({bold_text(tree.__class__.__name__)})"
    recurse(tree=tree, depth=1, prev_id=cur_id)
    return fmt.expandtabs(4)
