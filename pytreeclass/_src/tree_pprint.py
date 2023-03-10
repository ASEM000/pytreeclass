from __future__ import annotations

import ctypes
import functools as ft
import inspect
import math
import sys
from itertools import chain
from types import FunctionType
from typing import Any, Callable, Literal, Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax._src.custom_derivatives import custom_jvp
from jaxlib.xla_extension import CompiledFunction

import pytreeclass as pytc
from pytreeclass._src.tree_decorator import _dataclass_like_fields, _is_dataclass_like
from pytreeclass._src.tree_freeze import is_frozen, unfreeze
from pytreeclass._src.tree_trace import LeafTrace, tree_leaves_with_trace

PyTree = Any
Kind = Literal["repr", "str"]


def _node_pprint(node: Any, depth: int, kind: Kind, width: int) -> str:
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
    if isinstance(node, tuple) and hasattr(node, "_fields"):
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


def _general_pprint(node: Any, depth: int, kind: Kind, width: int) -> str:
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
    node: jax.ShapeDtypeStruct, depth: int, kind: Kind, width: int
) -> str:
    """Pretty print jax.ShapeDtypeStruct"""
    del depth, kind

    shape = (
        f"{node.shape}".replace(",", "")
        .replace("(", "[")
        .replace(")", "]")
        .replace(" ", ",")
        .replace("[]", "[0]")
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
    node: np.ndarray | jnp.ndarray, depth: int, kind: Kind, width: int
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

    # handle interval
    low, high = np.min(node), np.max(node)
    interval = "(" if math.isinf(low) else "["
    is_integer = issubclass(node.dtype.type, np.integer)
    interval += f"{low},{high}" if is_integer else f"{low:.2f},{high:.2f}"
    interval += ")" if math.isinf(high) else "]"  # resolve closed/open interval
    interval = interval.replace("inf", "∞")  # replace inf with infinity symbol

    # handle mean and std
    mean, std = f"{np.mean(node):.2f}", f"{np.std(node):.2f}"

    return _format_width(f"{base}(μ={mean}, σ={std}, ∈{interval})", width)


@ft.lru_cache
def _func_pprint(func: Callable, depth: int, kind: Kind, width: int) -> str:
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


def _slice_pprint(node: slice, depth: int, kind: Kind, width: int) -> str:
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


def _list_pprint(node: list, depth: int, kind: Kind, width: int) -> str:
    """Pretty print a list"""
    fmt = (f"{(_node_pprint(v,depth+1,kind,width))}" for v in node)
    fmt = (", \n" + "\t" * (depth + 1)).join(fmt)
    fmt = "[\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + "]"
    return _format_width(fmt, width)


def _tuple_pprint(node: tuple, depth: int, kind: Kind, width: int) -> str:
    """Pretty print a list"""
    fmt = (f"{(_node_pprint(v,depth+1,kind,width))}" for v in node)
    fmt = (", \n" + "\t" * (depth + 1)).join(fmt)
    fmt = "(\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + ")"
    return _format_width(fmt, width)


def _set_pprint(node: set, depth: int, kind: Kind, width: int) -> str:
    """Pretty print a list"""
    fmt = (f"{(_node_pprint(v,depth+1,kind,width))}" for v in node)
    fmt = (", \n" + "\t" * (depth + 1)).join(fmt)
    fmt = "{\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + "}"
    return _format_width(fmt, width)


def _dict_pprint(node: dict, depth: int, kind: Kind, width: int) -> str:
    fmt = (f"{k}:{_node_pprint(v,depth+1,kind,width)}" for k, v in node.items())
    fmt = (", \n" + "\t" * (depth + 1)).join(fmt)
    fmt = "{\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + "}"
    return _format_width(fmt, width)


def _namedtuple_pprint(node, depth: int, kind: Kind, width: int) -> str:
    items = node._asdict().items()
    fmt = (f"{k}={_node_pprint(v,depth+1,kind,width)}" for k, v in items)
    fmt = (", \n" + "\t" * (depth + 1)).join(fmt)
    fmt = "namedtuple(\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + ")"
    return _format_width(fmt, width)


def _dataclass_like_pprint(node, depth: int, kind: Kind, width: int) -> str:
    name = node.__class__.__name__
    fields = _dataclass_like_fields(node)
    # we use vars here to avoid unfreezing it in case it is frozen
    vs = (vars(node)[f.name] for f in fields if f.repr)
    fs = (f for f in fields if f.repr)
    fmt = (f"{f.name}={_node_pprint(v,depth+1,kind,width)}" for f, v in zip(fs, vs))
    fmt = (", \n" + "\t" * (depth + 1)).join(fmt)
    fmt = f"{name}(\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + ")"
    return _format_width(fmt, width)


def _node_type_pprint(node: type, depth: int, kind: Kind, width: int) -> str:
    if hasattr(node, "dtype") and hasattr(node, "shape"):
        shape_dype = node.shape, node.dtype
        fmt = _node_pprint(jax.ShapeDtypeStruct(*shape_dype), depth, kind, width)
    else:
        fmt = f"{node.__class__.__name__}"
    return _format_width(fmt, width)


def _should_omit_trace(trace: LeafTrace) -> bool:
    for metadata in trace.metadatas:
        if isinstance(metadata, dict) and "repr" in metadata:
            if metadata["repr"] is False:
                return True
    return False


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


def _resolve_names(trace: LeafTrace, width: int) -> str:
    # given a trace with a tuple of names, we resolve the names
    # to a single string
    path = trace.names[0]
    for name in trace.names[1:]:
        path += "" if name.startswith("[") else "."
        path += _node_pprint(name, 0, "str", width)
    return path


def tree_diagram(tree, depth: int | None = None, width: int = 60):
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
    if not (isinstance(depth, int) or depth is None):
        raise TypeError(f"depth must be an integer or `None`, got {type(depth)}")

    traces, leaves = zip(*tree_leaves_with_trace(tree, is_leaf=is_frozen, depth=depth))

    fmt = f"{type(tree).__name__}"

    for i, (trace, leaf) in enumerate(zip(traces, leaves)):
        if _should_omit_trace(trace):
            continue

        for depth, (name, type_) in enumerate(zip(trace.names, trace.types)):
            # skip printing the common parent node twice
            if i > 0 and traces[i - 1].names[: depth + 1] == trace.names[: depth + 1]:
                continue

            fmt += "\n\t"

            for di in range(depth):
                # handle printing the left lines for each depth
                if trace.indices[di][0] == trace.indices[di][1] - 1:
                    # do not print the left line
                    # └── A
                    #     └── B
                    fmt += " \t"
                else:
                    # print the left line
                    # ├── A
                    # │   └── B
                    # └── C
                    fmt += "│\t"

            if trace.indices[depth][0] == (trace.indices[depth][1] - 1):
                # check if we are at the last node in the current depth
                fmt += "└"
            else:
                fmt += "├"

            fmt += f"── {_node_pprint(name,0,'str',width )}"

            if depth == len(trace.names) - 1:
                # if we are at the leaf node, print the value as `=value`
                fmt += f"={_node_pprint(leaf,depth+2,'repr',width)}"
            else:
                # if we are not at the leaf node, print the type as `:type`
                fmt += f":{type_.__name__}"

    return fmt.expandtabs(4)


def tree_mermaid(tree: PyTree, depth=None, width: int = 60) -> str:
    # def _generate_mermaid_link(mermaid_string: str) -> str:
    #     """generate a one-time link mermaid diagram"""
    #     url_val = "https://pytreeclass.herokuapp.com/generateTemp"
    #     request = requests.post(url_val, json={"description": mermaid_string})
    #     generated_id = request.json()["id"]
    #     generated_html = f"https://pytreeclass.herokuapp.com/temp/?id={generated_id}"
    #     return f"Open URL in browser: {generated_html}"

    """generate a mermaid diagram syntax of a pytree"""
    if not (isinstance(depth, int) or depth is None):
        raise TypeError(f"depth must be an integer or `None`, got {type(depth)}")

    def bold_text(text: str) -> str:
        # bold a text in ansci code
        return "<b>" + text + "</b>"

    def node_id(input):
        """hash a value by its location in a tree. used to connect values in mermaid"""
        return ctypes.c_size_t(hash(input)).value

    traces, leaves = zip(*tree_leaves_with_trace(tree, is_leaf=is_frozen, depth=depth))
    # in case of a single node tree or depth=0, avoid printing the node twice
    # once for the trace and once for the summary

    root_id = node_id((0, 0, -1, 0))
    fmt = f"flowchart LR\n\tid{root_id}({bold_text(tree.__class__.__name__)})"
    cur_id = None

    for trace, leaf in zip(traces, leaves):
        if _should_omit_trace(trace):
            continue

        count, size = _calculate_leaf_trace_stats(trace, leaf)
        count = _format_count(count) + " leaf"
        size = _format_size(size)

        for depth, (name, type_) in enumerate(zip(trace.names, trace.types)):
            name = _node_pprint(name, 0, "str", width)

            prev_id = root_id if depth == 0 else cur_id
            cur_id = node_id((depth, tuple(trace.indices), prev_id))
            fmt += f"\n\tid{prev_id}"
            stats = f'|"{count}<br>{size}"|' if depth == len(trace.names) - 1 else ""
            fmt += "--->" + stats
            is_last = depth == len(trace.names) - 1
            value = f"={_node_pprint(leaf,0,'repr',width)}" if is_last else ""
            fmt += f'id{cur_id}("{bold_text(name)}:{type_.__name__}{value}")'

    return fmt.expandtabs(4)


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


def _calculate_leaf_trace_stats(
    trace: LeafTrace, tree: Any
) -> tuple[int | complex, int | complex]:
    # calcuate some stats of a single subtree defined by the `NodeInfo` objects
    # for each subtree, we will calculate the types distribution and their size
    # stats = defaultdict(lambda: [0, 0])
    if not isinstance(trace, LeafTrace):
        raise TypeError(f"Expected `LeafTrace` object, but got {type(trace)}")

    count = size = 0

    traces, leaves = zip(*tree_leaves_with_trace(tree, is_leaf=is_frozen))

    for trace, leaf in zip(traces, leaves):
        # unfrozen leaf
        leaf_ = unfreeze(leaf)
        # array count is the product of the shape. if the node is not an array, then the count is 1
        count_ = int(np.array(leaf_.shape).prod()) if hasattr(leaf_, "shape") else 1
        size_ = leaf_.nbytes if hasattr(leaf_, "nbytes") else sys.getsizeof(leaf_)

        if is_frozen(tree) or is_frozen(leaf):
            count_ = complex(0, count_)
            size_ = complex(0, size_)

        count += count_
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


def tree_summary(tree: PyTree, *, depth=None, width: int = 60) -> str:
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
        ┌─────────┬────┬─────┬──────┐
        │Name     │Type│Count│Size  │
        ├─────────┼────┼─────┼──────┤
        │[0]      │int │1    │28.00B│
        ├─────────┼────┼─────┼──────┤
        │[1][0]   │int │1    │28.00B│
        ├─────────┼────┼─────┼──────┤
        │[1][1][0]│int │1    │28.00B│
        ├─────────┼────┼─────┼──────┤
        │Σ        │list│3    │84.00B│
        └─────────┴────┴─────┴──────┘
    """
    if not (isinstance(depth, int) or depth is None):
        raise TypeError(f"depth must be an integer or `None`, got {type(depth)}")

    ROWS = [["Name", "Type", "Count", "Size"]]

    traces, leaves = zip(*tree_leaves_with_trace(tree, is_leaf=is_frozen, depth=depth))
    # in case of a single node tree or depth=0, avoid printing the node twice
    # once for the trace and once for the summary
    traces = traces if len(traces) > 1 else ()

    for trace, leaf in zip(traces, leaves):
        if _should_omit_trace(trace):
            continue

        row = [_resolve_names(trace, width)]

        # type name row
        row += [_node_type_pprint(pytc.unfreeze(leaf), 0, "str", width)]

        # count and size row
        count, size = _calculate_leaf_trace_stats(trace, leaf)
        leaves_count = _format_count(count.real + count.imag)
        leaves_size = _format_size(size.real + size.imag)

        # add frozen stats only if there are frozen leaves
        leaves_count += f"({_format_count(count.imag)})" if count.imag > 0 else ""
        leaves_size += f"({_format_size(size.imag)})" if size.imag > 0 else ""
        row += [leaves_count, leaves_size]

        ROWS += [row]

    COUNT = [complex(0), complex(0)]  # non-frozen, frozen
    SIZE = [complex(0), complex(0)]

    for trace, leaf in tree_leaves_with_trace(tree, is_leaf=is_frozen):
        count, size = _calculate_leaf_trace_stats(trace, leaf)
        COUNT[is_frozen(leaf)] += count
        SIZE[is_frozen(leaf)] += size

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
