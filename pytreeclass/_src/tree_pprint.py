from __future__ import annotations

import ctypes
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
from pytreeclass._src.tree_freeze import is_frozen
from pytreeclass._src.tree_viz_util import (
    _calculate_node_info_stats,
    _dataclass_like_fields,
    _format_count,
    _format_size,
    _format_width,
    _is_dataclass_like,
    _table,
    _tree_trace,
)

PyTree = Any
MAX_DEPTH = float("inf")


def _node_pprint(node: Any, depth: int, kind: str, width: int) -> str:
    if isinstance(node, ft.partial):
        # applies for partial functions including `jax.tree_util.Partial`
        return f"Partial({_func_pprint(node.func, depth,kind,width)})"
    if isinstance(node, (FunctionType, custom_jvp)):
        return _func_pprint(node, depth, kind, width)
    if isinstance(node, CompiledFunction):
        # special case for jitted JAX functions
        return (f"jit({_func_pprint(node)})", depth, kind, width)
    if isinstance(node, (np.ndarray, jnp.ndarray)):
        return _numpy_pprint(node, depth, kind, width)
    if isinstance(node, jax.ShapeDtypeStruct):
        return _shape_dtype_struct_pprint(node, depth, kind, width)
    if hasattr(node, "_fields") and hasattr(node, "_asdict"):
        return _namedtuple_pprint(node, depth, kind, width)
    if isinstance(node, list):
        return _list_pprint(node, depth, kind, width)
    if isinstance(node, tuple):
        return _tuple_pprint(node, depth, kind, width)
    if isinstance(node, set):
        return _set_pprint(node, depth, kind, width)
    if isinstance(node, dict):
        return _dict_pprint(node, depth, kind, width)
    if _is_dataclass_like(node):
        return _dataclass_like_pprint(node, depth, kind, width)
    if isinstance(node, slice):
        return _slice_pprint(node, depth, kind, width)
    if is_frozen(node):
        return f"#{_node_pprint(node.unwrap(), depth, kind, width)}"
    return _general_pprint(node, depth, kind, width)


def _general_pprint(node: Any, depth: int, kind: str, width: int) -> str:
    if isinstance(node, object) and node.__class__.__repr__ is not object.__repr__:
        # use custom __repr__ method if available
        fmt = f"{node!r}" if kind == "repr" else f"{node!s}"
    elif isinstance(node, type):
        fmt = f"{node!r}" if kind == "repr" else f"{node!s}"
    else:
        # use `jax.tree_util.tree_map`, to get representation of the node
        leaves = enumerate(jtu.tree_leaves(node))
        fmt = (f"leaf_{i}={_node_pprint(v,depth,kind,width)}" for i, v in (leaves))
        fmt = ", ".join(fmt)
        fmt = f"{node.__class__.__name__}({fmt})"

    is_mutltiline = "\n" in fmt

    # multiline repr/str case, increase depth and indent
    depth = (depth + 1) if is_mutltiline else depth
    fmt = ("\n" + "\t" * depth).join(fmt.split("\n"))
    fmt = ("\n" + "\t" * (depth) + fmt) if is_mutltiline else fmt
    return _format_width(fmt, width)


def _shape_dtype_struct_pprint(
    node: jax.ShapeDtypeStruct, depth: int, kind: str, width: int
) -> str:
    """Pretty print jax.ShapeDtypeStruct"""
    del depth, kind

    shape = (
        f"{node.shape}".replace(",", "")
        .replace("(", "{")
        .replace(")", "}")
        .replace(" ", "x")
        .replace("{}", "{0}")
    )
    if issubclass(node.dtype.type, np.integer):
        fmt = f"{node.dtype}".replace("int", "i") + shape
    elif issubclass(node.dtype.type, np.floating):
        fmt = f"{node.dtype}".replace("float", "f") + shape
    elif issubclass(node.dtype.type, np.complexfloating):
        fmt = f"{node.dtype}".replace("complex", "c") + shape
    else:
        fmt = f"{node.dtype}" + shape
    return _format_width(fmt, width)


def _numpy_pprint(
    node: np.ndarray | jnp.ndarray, depth: int, kind: str, width: int
) -> str:
    """Replace np.ndarray repr with short hand notation for type and shape"""
    if kind == "str":
        return _general_pprint(node, depth, "str", width)

    base = _shape_dtype_struct_pprint(node, depth, kind, width)

    if not issubclass(node.dtype.type, (np.integer, np.floating)) or node.size == 0:
        return _format_width(base, width)

    # Extended repr for numpy array, with extended information
    # this part of the function is inspired by
    # lovely-jax https://github.com/xl0/lovely-jax

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

    # add brackets to indicate closed/open interval
    interval += ")" if math.isinf(high) else "]"
    # replace inf with infinity symbol
    interval = interval.replace("inf", "∞")
    # return extended repr
    return _format_width(f"{base}∈{interval}", width)


@ft.lru_cache
def _func_pprint(func: Callable, depth: int, kind: str, width: int) -> str:
    """Pretty print function

    Example:
        >>> def example(a: int, b=1, *c, d, e=2, **f) -> str:
            ...
        >>> _func_pprint(example)
        "example(a, b, *c, d, e, **f)"
    """
    del depth, kind

    args, varargs, varkw, _, kwonlyargs, _, _ = inspect.getfullargspec(func)
    args = (", ".join(args)) if len(args) > 0 else ""
    varargs = ("*" + varargs) if varargs is not None else ""
    kwonlyargs = (", ".join(kwonlyargs)) if len(kwonlyargs) > 0 else ""
    varkw = ("**" + varkw) if varkw is not None else ""
    name = "Lambda" if (func.__name__ == "<lambda>") else func.__name__
    fmt = f"{name}("
    fmt += ", ".join(item for item in [args, varargs, kwonlyargs, varkw] if item != "")
    fmt += ")"
    return _format_width(fmt, width)


def _slice_pprint(node: slice, depth: int, kind: str, width: int) -> str:
    del depth, kind
    start = node.start if node.start is not None else ""
    stop = node.stop if node.stop is not None else ""
    step = node.step if node.step is not None else ""

    if step and step > 1:
        fmt = f"[{start}:{stop}:{step}]"
    elif stop == start + 1:
        fmt = f"[{start}]"
    else:
        fmt = f"[{start}:{stop}]"
    return _format_width(fmt, width)


def _list_pprint(node: list, depth: int, kind: str, width: int) -> str:
    """Pretty print a list"""
    fmt = (f"{(_node_pprint(v,depth+1,kind,width))}" for v in node)
    fmt = (", \n" + "\t" * (depth + 1)).join(fmt)
    fmt = "[\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + "]"
    return _format_width(fmt, width)


def _tuple_pprint(node: tuple, depth: int, kind: str, width: int) -> str:
    """Pretty print a list"""
    fmt = (f"{(_node_pprint(v,depth+1,kind,width))}" for v in node)
    fmt = (", \n" + "\t" * (depth + 1)).join(fmt)
    fmt = "(\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + ")"
    return _format_width(fmt, width)


def _set_pprint(node: set, depth: int, kind: str, width: int) -> str:
    """Pretty print a list"""
    fmt = (f"{(_node_pprint(v,depth+1,kind,width))}" for v in node)
    fmt = (", \n" + "\t" * (depth + 1)).join(fmt)
    fmt = "{\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + "}"
    return _format_width(fmt, width)


def _dict_pprint(node: dict, depth: int, kind: str, width: int) -> str:
    fmt = (f"{k}:{_node_pprint(v,depth+1,kind,width)}" for k, v in node.items())
    fmt = (", \n" + "\t" * (depth + 1)).join(fmt)
    fmt = "{\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + "}"
    return _format_width(fmt, width)


def _namedtuple_pprint(node, depth: int, kind: str, width: int) -> str:
    items = node._asdict().items()
    fmt = (f"{k}={_node_pprint(v,depth+1,kind,width)}" for k, v in items)
    fmt = (", \n" + "\t" * (depth + 1)).join(fmt)
    fmt = "namedtuple(\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + ")"
    return _format_width(fmt, width)


def _dataclass_like_pprint(node, depth: int, kind: str, width: int) -> str:
    name = node.__class__.__name__
    fields = _dataclass_like_fields(node)
    # we use vars here to avoid unfreezing it in case it is frozen
    vs = (vars(node)[f.name] for f in fields if f.repr)
    fs = (f for f in fields if f.repr)
    fmt = (f"{f.name}={_node_pprint(v,depth+1,kind,width)}" for f, v in zip(fs, vs))
    fmt = (", \n" + "\t" * (depth + 1)).join(fmt)
    fmt = f"{name}(\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + ")"
    return _format_width(fmt, width)


def _node_type_pprint(node: type, depth: int, kind: str, width: int) -> str:
    if hasattr(node, "dtype") and hasattr(node, "shape"):
        shape_dype = node.shape, node.dtype
        fmt = _node_pprint(jax.ShapeDtypeStruct(*shape_dype), depth, kind, width)
    else:
        fmt = f"{node.__class__.__name__}"
    return _format_width(fmt, width)


def tree_repr(tree: PyTree, *, width: int = 80, indent: int = 2) -> str:
    """Prertty print dataclass tree `__repr__`

    Args:
        tree: PyTree
        width: max width of the repr string
        indent: indent size
    """
    return _node_pprint(tree, 0, kind="repr", width=width).expandtabs(indent)


def tree_str(tree: PyTree, *, width: int = 80, indent: int = 2) -> str:
    """Prertty print dataclass tree `__str__`

    Args:
        tree: PyTree
        width: max width of the str string
        indent: indent size
    """
    return _node_pprint(tree, depth=0, kind="str", width=width).expandtabs(indent)


def tree_diagram(tree, depth: int = MAX_DEPTH, width: int = 60):
    """Pretty print treeclass tree with tree structure diagram

    Args:
        tree: PyTree
        depth: depth of the tree to print. default is max depth
        width: max width of the str string

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
    """
    if not (isinstance(depth, int) or depth is MAX_DEPTH):
        raise TypeError(f"depth must be an integer, got {type(depth)}")

    infos = _tree_trace(tree, depth)
    fmt = f"{tree.__class__.__name__}"

    for i, info in enumerate(infos):
        # iterate over the leaves `NodeInfo` object
        if not info.repr:
            continue

        max_depth = len(info.names)
        index = info.index

        for depth, (name, type_) in enumerate(zip(info.names, info.types)):
            # skip printing the common parent node twice
            if i > 0 and infos[i - 1].names[: depth + 1] == info.names[: depth + 1]:
                continue

            is_last = depth == max_depth - 1

            fmt += "\n\t"
            fmt += "".join(("" if index[i] == 0 else "│") + "\t" for i in range(depth))
            fmt += "├" if not index[depth] == 0 else "└"
            mark = "#" if (info.frozen and is_last) else "─"
            fmt += f"{mark}─ {_node_pprint(name,0,'str',width )}"
            fmt += (
                f"={_node_pprint(info.node,depth+2,'repr',width)}"
                if is_last
                else f":{type_.__name__}"
            )

    return fmt.expandtabs(4)


def tree_summary(tree: PyTree, *, depth=MAX_DEPTH, width: int = 60) -> str:
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
    if not (isinstance(depth, int) or depth is MAX_DEPTH):
        raise TypeError(f"depth must be an integer, got {type(depth)}")

    ROWS = [["Name", "Type", "Count", "Size"]]

    # traverse the tree and collect info about each node
    infos = _tree_trace(tree, depth)
    # in case of a single node tree or depth=0, avoid printing the node twice
    # once for the trace and once for the summary
    infos = infos if len(infos) > 1 else ()

    for info in infos:
        if not info.repr:
            # skip nodes that are not repr
            continue

        path = ".".join(_node_pprint(i, 0, "str", width) for i in info.names)
        row = [path.replace("].", "]").replace(".[", "[")]

        # type name row
        row += [_node_type_pprint(pytc.unfreeze(info.node), 0, "str", width)]

        # count and size row
        count, size = _calculate_node_info_stats(info)
        leaves_count = _format_count(count.real + count.imag)
        leaves_size = _format_size(size.real + size.imag)

        # add frozen stats only if there are frozen leaves
        leaves_count += f"({_format_count(count.imag)})" if count.imag > 0 else ""
        leaves_size += f"({_format_size(size.imag)})" if size.imag > 0 else ""
        row += [leaves_count, leaves_size]

        ROWS += [row]

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
    row += [_node_type_pprint(tree, 0, "repr", width)]
    total_count = _format_count(total_count)

    total_size = _format_size(total_size)

    if frozen_count > 0:
        total_count += f"({_format_count(frozen_count)})"
        total_size += f"({_format_size(frozen_size)})"
        # add frozen to the header row if there are frozen leaves
        # otherwise donot bloat the header row
        ROWS[0][2] = ROWS[0][2] + "(Frozen)"
        ROWS[0][3] = ROWS[0][3] + "(Frozen)"

    row += [total_count, total_size]
    ROWS += [row]

    COLS = [list(c) for c in zip(*ROWS)]
    layer_table = _table(COLS)
    return layer_table.expandtabs(8)


def tree_mermaid(tree: PyTree, depth=MAX_DEPTH, width: int = 60) -> str:
    # def _generate_mermaid_link(mermaid_string: str) -> str:
    #     """generate a one-time link mermaid diagram"""
    #     url_val = "https://pytreeclass.herokuapp.com/generateTemp"
    #     request = requests.post(url_val, json={"description": mermaid_string})
    #     generated_id = request.json()["id"]
    #     generated_html = f"https://pytreeclass.herokuapp.com/temp/?id={generated_id}"
    #     return f"Open URL in browser: {generated_html}"

    """generate a mermaid diagram syntax of a pytree"""
    if not (isinstance(depth, int) or depth is MAX_DEPTH):
        raise TypeError(f"depth must be an integer, got {type(depth)}")

    def bold_text(text: str) -> str:
        # bold a text in ansci code
        return "<b>" + text + "</b>"

    def node_id(input):
        """hash a value by its location in a tree. used to connect values in mermaid"""
        return ctypes.c_size_t(hash(input)).value

    infos = _tree_trace(tree, depth)

    root_id = node_id((0, 0, -1, 0))
    fmt = f"flowchart LR\n\tid{root_id}({bold_text(tree.__class__.__name__)})"
    cur_id = None

    for info in infos:
        if not info.repr:
            continue

        count, size = _calculate_node_info_stats(info)
        count = _format_count(count) + " leaf"
        size = _format_size(size)

        for depth, (name, type_) in enumerate(zip(info.names, info.types)):
            name = _node_pprint(name, 0, "str", width)
            prev_id = root_id if depth == 0 else cur_id
            cur_id = node_id((depth, tuple(info.index), prev_id))
            mark = "-..-" if info.frozen else "--->"
            fmt += f"\n\tid{prev_id}"
            stats = f'|"{count}<br>{size}"|' if depth == len(info.names) - 1 else ""
            fmt += f"{mark}" + stats
            is_last = depth == len(info.names) - 1
            value = f"={_node_pprint(info.node,0,'repr',width)}" if is_last else ""
            fmt += f'id{cur_id}("{bold_text(name)}:{type_.__name__}{value}")'

    return fmt.expandtabs(4)
