from __future__ import annotations

import ctypes
import dataclasses as dc
import functools as ft
import inspect
import math
import sys
from itertools import chain
from types import FunctionType
from typing import Any, Callable, Literal

import jax
import numpy as np
from jax._src.custom_derivatives import custom_jvp
from jaxlib.xla_extension import PjitFunction

import pytreeclass as pytc

PyTree = Any
PrintKind = Literal["repr", "str"]


def _node_pprint(
    node: Any, indent: int, kind: PrintKind, width: int, depth: int | float
) -> str:
    if depth < 0:
        return "..."
    if isinstance(node, ft.partial):
        return f"Partial({_func_pprint(node.func, indent,kind,width,depth)})"
    if isinstance(node, (FunctionType, custom_jvp)):
        return _func_pprint(node, indent, kind, width, depth)
    if isinstance(node, PjitFunction):
        return f"jit({_func_pprint(node, indent, kind, width,depth)})"
    if isinstance(node, (np.ndarray, jax.Array)):
        return _numpy_pprint(node, indent, kind, width, depth)
    if isinstance(node, jax.ShapeDtypeStruct):
        return _shape_dtype_pprint(node, indent, kind, width, depth)
    if isinstance(node, tuple) and hasattr(node, "_fields"):
        return _namedtuple_pprint(node, indent, kind, width, depth)
    if isinstance(node, list):
        return _list_pprint(node, indent, kind, width, depth)
    if isinstance(node, tuple):
        return _tuple_pprint(node, indent, kind, width, depth)
    if isinstance(node, set):
        return _set_pprint(node, indent, kind, width, depth)
    if isinstance(node, dict):
        return _dict_pprint(node, indent, kind, width, depth)
    if dc.is_dataclass(node):
        return _dataclass_pprint(node, indent, kind, width, depth)
    if pytc.is_treeclass(node):
        return _treeclass_pprint(node, indent, kind, width, depth)
    if pytc.is_frozen(node):
        return f"#{_node_pprint(node.unwrap(), indent, kind, width,depth)}"
    return _general_pprint(node, indent, kind, width, depth)


def _general_pprint(
    node: Any, indent: int, kind: PrintKind, width: int, depth: int
) -> str:
    del depth

    fmt = f"{node!r}" if kind == "repr" else f"{node!s}"
    is_mutltiline = "\n" in fmt

    # multiline repr/str case, increase indent and indent
    indent = (indent + 1) if is_mutltiline else indent
    fmt = ("\n" + "\t" * indent).join(fmt.split("\n"))
    fmt = ("\n" + "\t" * (indent) + fmt) if is_mutltiline else fmt
    return _format_width(fmt, width)


def _shape_dtype_pprint(
    node: Any, indent: int, kind: PrintKind, width: int, depth: int
) -> str:
    """Pretty print a node with dtype and shape"""
    del indent, kind, depth

    shape = f"{node.shape}".replace(",", "")
    shape = shape.replace("(", "[")
    shape = shape.replace(")", "]")
    shape = shape.replace(" ", ",")
    shape = shape.replace("[]", "[0]")
    dtype = f"{node.dtype}".replace("int", "i")
    dtype = dtype.replace("float", "f")
    dtype = dtype.replace("complex", "c")
    return _format_width(dtype + shape, width)


def _numpy_pprint(
    node: np.ndarray | jax.Array, indent: int, kind: PrintKind, width: int, depth: int
) -> str:
    """Replace np.ndarray repr with short hand notation for type and shape"""
    if kind == "str":
        return _general_pprint(node, indent, "str", width, depth)

    base = _shape_dtype_pprint(node, indent, kind, width, depth)

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
def _func_pprint(
    func: Callable, indent: int, kind: PrintKind, width: int, depth: int
) -> str:
    # Pretty print function
    # Example:
    #     >>> def example(a: int, b=1, *c, d, e=2, **f) -> str:
    #         ...
    #     >>> _func_pprint(example)
    #     "example(a, b, *c, d, e, **f)"
    del indent, kind, depth

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


def _list_pprint(
    node: list, indent: int, kind: PrintKind, width: int, depth: int
) -> str:
    fmt = (f"{(_node_pprint(v,indent+1,kind,width,depth-1))}" for v in node)
    fmt = (", \n" + "\t" * (indent + 1)).join(fmt)
    fmt = "[\n" + "\t" * (indent + 1) + (fmt) + "\n" + "\t" * (indent) + "]"
    return _format_width(fmt, width)


def _tuple_pprint(
    node: tuple, indent: int, kind: PrintKind, width: int, depth: int
) -> str:
    fmt = (f"{(_node_pprint(v,indent+1,kind,width,depth-1))}" for v in node)
    fmt = (", \n" + "\t" * (indent + 1)).join(fmt)
    fmt = "(\n" + "\t" * (indent + 1) + (fmt) + "\n" + "\t" * (indent) + ")"
    return _format_width(fmt, width)


def _set_pprint(node: set, indent: int, kind: PrintKind, width: int, depth: int) -> str:
    fmt = (f"{(_node_pprint(v,indent+1,kind,width,depth-1))}" for v in node)
    fmt = (", \n" + "\t" * (indent + 1)).join(fmt)
    fmt = "{\n" + "\t" * (indent + 1) + (fmt) + "\n" + "\t" * (indent) + "}"
    return _format_width(fmt, width)


def _dict_pprint(
    node: dict, indent: int, kind: PrintKind, width: int, depth: int
) -> str:
    fmt = (f"{k}:{_node_pprint(v,indent+1,kind,width,depth-1)}" for k, v in node.items())  # fmt: skip
    fmt = (", \n" + "\t" * (indent + 1)).join(fmt)
    fmt = "{\n" + "\t" * (indent + 1) + (fmt) + "\n" + "\t" * (indent) + "}"
    return _format_width(fmt, width)


def _namedtuple_pprint(
    node: Any, indent: int, kind: PrintKind, width: int, depth: int
) -> str:
    items = node._asdict().items()
    fmt = (f"{k}={_node_pprint(v,indent+1,kind,width,depth-1)}" for k, v in items)
    fmt = (", \n" + "\t" * (indent + 1)).join(fmt)
    fmt = "namedtuple(\n" + "\t" * (indent + 1) + (fmt) + "\n" + "\t" * (indent) + ")"
    return _format_width(fmt, width)


def _dataclass_pprint(
    node: Any, indent: int, kind: PrintKind, width: int, depth: int
) -> str:
    name = type(node).__name__
    fields = dc.fields(node)
    vs = (vars(node)[F.name] for F in fields if F.repr)
    fs = (F for F in fields if F.repr)
    fmt = (f"{f.name}={_node_pprint(v,indent+1,kind,width,depth-1)}"for f, v in zip(fs, vs))  # fmt: skip
    fmt = (", \n" + "\t" * (indent + 1)).join(fmt)
    fmt = f"{name}(\n" + "\t" * (indent + 1) + (fmt) + "\n" + "\t" * (indent) + ")"
    return _format_width(fmt, width)


def _treeclass_pprint(
    node: Any, indent: int, kind: PrintKind, width: int, depth: int
) -> str:
    name = type(node).__name__
    fields = pytc.fields(node)
    vs = (vars(node)[F.name] for F in fields if F.repr)
    fs = (F for F in fields if F.repr)
    fmt = (f"{f.name}={_node_pprint(v,indent+1,kind,width,depth-1)}"for f, v in zip(fs, vs))  # fmt: skip
    fmt = (", \n" + "\t" * (indent + 1)).join(fmt)
    fmt = f"{name}(\n" + "\t" * (indent + 1) + (fmt) + "\n" + "\t" * (indent) + ")"
    return _format_width(fmt, width)


def _node_type_pprint(
    node: jax.Array | np.ndarray, indent: int, kind: PrintKind, width: int, depth: int
) -> str:
    if isinstance(node, (jax.Array, np.ndarray)):
        shape_dype = node.shape, node.dtype
        fmt = _node_pprint(jax.ShapeDtypeStruct(*shape_dype), indent, kind, width, depth)  # fmt: skip
    else:
        fmt = f"{type(node).__name__}"
    return _format_width(fmt, width)


def _should_omit_trace(metadatas) -> bool:
    for metadata in metadatas:
        if isinstance(metadata, dict) and "repr" in metadata:
            if metadata["repr"] is False:
                return True
    return False


def tree_repr(
    tree: PyTree,
    *,
    width: int = 80,
    tabwidth: int = 2,
    depth: int | float = float("inf"),
) -> str:
    """Prertty print arbitrary PyTrees `__repr__`

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
        {a:..., b:..., c:..., f:...}

        >>> print(pytc.tree_repr(tree, depth=1))
        {a:1, b:[..., ...], c:{d:..., e:...}, f:i32[2](μ=6.50, σ=0.50, ∈[6,7])}

        >>> print(pytc.tree_repr(tree, depth=2))
        {a:1, b:[2, 3], c:{d:4, e:5}, f:i32[2](μ=6.50, σ=0.50, ∈[6,7])}

        >>> print(pytc.tree_repr(tree, tabwidth=8, width=20))
        {
                a:1,
                b:[2, 3],
                c:{d:4, e:5},
                f:i32[2](μ=6.50, σ=0.50, ∈[6,7])
        }
    """
    return _node_pprint(tree, 0, "repr", width, depth).expandtabs(tabwidth)


def tree_str(
    tree: PyTree,
    *,
    width: int = 80,
    tabwidth: int = 2,
    depth: int | float = float("inf"),
) -> str:
    """Prertty print arbitrary PyTrees `__str__`

    Args:
        tree: PyTree
        width: max width of the str string
        tabwidth: tab width of the repr string
        depth: max depth of the repr string
    Example:
        >>> import pytreeclass as pytc
        >>> import jax.numpy as jnp
        >>> tree = {'a' : 1, 'b' : [2, 3], 'c' : {'d' : 4, 'e' : 5} , 'f' : jnp.array([6, 7])}

        >>> print(pytc.tree_str(tree, depth=0))
        {a:..., b:..., c:..., f:...}

        >>> print(pytc.tree_str(tree, depth=1))
        {a:1, b:[..., ...], c:{d:..., e:...}, f:[6 7]}

        >>> print(pytc.tree_str(tree, depth=2))
        {a:1, b:[2, 3], c:{d:4, e:5}, f:[6 7]}

        >>> print(pytc.tree_str(tree, tabwidth=8, width=20))
        {
                a:1,
                b:[2, 3],
                c:{d:4, e:5},
                f:[6 7]
        }
    """
    return _node_pprint(tree, 0, "str", width, depth).expandtabs(tabwidth)


def _resolve_names(names, width: int) -> str:
    # given a trace with a tuple of names, we resolve the names
    # to a single string
    path = names[0]
    for name in names[1:]:
        path += "" if name.startswith("[") else "."
        path += _node_pprint(name, 0, "str", width, float("inf"))
    return path


def _is_trace_leaf_depth_factory(depth: int):
    # generate `is_trace_leaf` function to stop tracing at a certain `depth`
    # in essence, depth is the length of the trace entry
    def is_trace_leaf(trace) -> bool:
        # trace is a tuple of (names, leaves, tracers, aux_data)
        # done like this to ensure 4-tuple unpacking
        names, _, __, ___ = trace
        # stop tracing if depth is reached
        return False if depth is None else (depth < len(names))

    return is_trace_leaf


def _sibling_nodes_count_at_all_depth(lhs_trace, traces: tuple[Any]) -> list[int]:
    # given a trace and a list of traces, we count the number of nodes
    # at each depth that are siblings of the lhs_trace
    def sibling_nodes_count_at_depth(lhs_trace: Any, traces: tuple[Any], depth: int):
        result = set()
        start = False
        for trace in traces:
            _, __, indices, ___ = trace
            if len(indices) > depth and indices[:depth] == lhs_trace[2][:depth]:
                start = True
                # mere existence of a name at a given depth means
                # that there is a node at that depth
                result.add(indices[: depth + 1])
            elif start is True:
                # we already found the first sibling, so if we are here
                # it means that we have reached the end of the siblings
                break
        return len(result)

    depth, result = 0, []
    while True:
        if (out := sibling_nodes_count_at_depth(lhs_trace, traces, depth=depth)) == 0:
            break
        else:
            result += [out]
            depth += 1
    return result


def tree_diagram(tree, *, width: int = 60, depth: int | float = float("inf")):
    """Pretty print arbitrary PyTrees tree with tree structure diagram.

    Args:
        tree: PyTree
        depth: depth of the tree to print. default is max depth
        width: max width of the str string

    Example:
        >>> import pytreeclass as pytc
        >>> @pytc.treeclass
        ... class A:
        ...        x: int = 10
        ...        y: int = (20,30)
        ...        z: int = 40

        >>> @pytc.treeclass
        ... class B:
        ...     a: int = 10
        ...     b: tuple = (20,30, A())

        >>> print(pytc.tree_diagram(B(), depth=0))
        B

        >>> print(pytc.tree_diagram(B(), depth=1))
        B
            ├── a=10
            └── b=(..., ..., ...)


        >>> print(pytc.tree_diagram(B(), depth=2))
        B
            ├── a=10
            └── b:tuple
                ├── [0]=20
                ├── [1]=30
                └── [2]=A(x=10, y=(..., ...), z=40)
    """
    traces, leaves = zip(
        *pytc.tree_leaves_with_trace(
            tree=tree,
            is_leaf=pytc.is_frozen,
            is_trace_leaf=_is_trace_leaf_depth_factory(depth),
        )
    )

    fmt = f"{type(tree).__name__}"

    for i, (trace, leaf) in enumerate(zip(traces, leaves)):
        # i iterates over traces
        names, types, indices, metadatas = trace

        if _should_omit_trace(metadatas):
            continue

        sibling_nodes_count = _sibling_nodes_count_at_all_depth(trace, traces)

        for j, (name, type_) in enumerate(zip(names, types)):
            # j iterates over the depth of each trace
            if j == 0:
                # skip printing the root node
                continue

            # skip printing the common parent node twice
            prev_names, _, __, ___ = traces[i - 1]

            if i > 0 and prev_names[: j + 1] == names[: j + 1]:
                continue

            fmt += "\n\t"

            for k in range(j):
                if k == 0:
                    # skip printing the root node
                    continue

                # handle printing the left lines for each depth
                if indices[k] == sibling_nodes_count[k] - 1:
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

            if indices[j] == (sibling_nodes_count[j] - 1):
                # check if we are at the last node in the current depth
                fmt += "└"
            else:
                fmt += "├"

            fmt += f"── {_node_pprint(name,0,'str',width, depth )}"

            if j == len(names) - 1:
                # if we are at the leaf node, print the value as `=value`
                fmt += f"={_node_pprint(leaf,j+1,'repr',width, depth-1)}"
            else:
                # if we are not at the leaf node, print the type as `:type`
                fmt += f":{type_.__name__}"

    return fmt.expandtabs(4)


def tree_mermaid(
    tree: PyTree, width: int = 60, depth: int | float = float("inf")
) -> str:
    # def _generate_mermaid_link(mermaid_string: str) -> str:
    #     """generate a one-time link mermaid diagram"""
    #     url_val = "https://pytreeclass.herokuapp.com/generateTemp"
    #     request = requests.post(url_val, json={"description": mermaid_string})
    #     generated_id = request.json()["id"]
    #     generated_html = f"https://pytreeclass.herokuapp.com/temp/?id={generated_id}"
    #     return f"Open URL in browser: {generated_html}"

    """generate a mermaid diagram syntax for arbitrary PyTrees."""

    def bold_text(text: str) -> str:
        # bold a text in ansci code
        return "<b>" + text + "</b>"

    def node_id(input):
        # hash a value by its location in a tree. used to connect values in mermaid
        # specifically we use c_size_t to avoid negative values in the hash that is not supported by mermaid
        return ctypes.c_size_t(hash(input)).value

    traces, leaves = zip(
        *pytc.tree_leaves_with_trace(
            tree=tree,
            is_leaf=pytc.is_frozen,
            is_trace_leaf=_is_trace_leaf_depth_factory(depth),
        )
    )
    # in case of a single node tree or depth=0, avoid printing the node twice
    # once for the trace and once for the summary

    root_id = node_id((0, 0, -1, 0))
    fmt = f"flowchart LR\n\tid{root_id}({bold_text(type(tree).__name__)})"
    cur_id = None

    for trace, leaf in zip(traces, leaves):
        names, types, indices, metadatas = trace

        if _should_omit_trace(metadatas):
            continue

        count, size = _calculate_leaf_trace_stats(leaf)
        count = _format_count(count) + " leaf"
        size = _format_size(size)

        for depth, (name, type_) in enumerate(zip(names, types)):
            if depth == 0:
                # skip printing the root node trace
                continue

            name = _node_pprint(name, 0, "str", width, depth)

            prev_id = root_id if depth == 1 else cur_id
            cur_id = node_id((depth - 1, tuple(indices[1:]), prev_id))
            fmt += f"\n\tid{prev_id}"
            stats = f'|"{count}<br>{size}"|' if depth == len(names) - 1 else ""
            fmt += "--->" + stats
            is_last = depth == len(names) - 1
            value = f"={_node_pprint(leaf,0,'repr',width,depth-1)}" if is_last else ""
            fmt += f'id{cur_id}("{bold_text(name)}:{type_.__name__}{value}")'

    return fmt.expandtabs(4)


def _format_width(string, width=60):
    """strip newline/tab characters if less than max width"""
    children_length = len(string) - string.count("\n") - string.count("\t")
    if children_length > width:
        return string
    return string.replace("\n", "").replace("\t", "")


def _format_size(node_size, newline=False):
    # return formatted size from inexact(exact) complex number
    # Examples:
    #     >>> _format_size(1024)
    #     '1.00KB'
    #     >>> _format_size(1024**2)
    #     '1.00MB'

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
    # return formatted count from inexact(exact) complex number
    # Examples:
    #     >>> _format_count(1024)
    #     '1,024'

    #     >>> _format_count(1024**2)
    #     '1,048,576'

    mark = "\n" if newline else ""

    if isinstance(node_count, complex):
        return f"{int(node_count.real):,}{mark}({int(node_count.imag):,})"

    if isinstance(node_count, (float, int)):
        return f"{int(node_count):,}"

    raise TypeError(f"node_count must be int or float, got {type(node_count)}")


def _calculate_leaf_trace_stats(tree: Any) -> tuple[int | complex, int | complex]:
    # calcuate some stats of a single subtree defined
    counts = sizes = 0
    _, leaves = zip(*pytc.tree_leaves_with_trace(tree, is_leaf=pytc.is_frozen))

    for leaf in leaves:
        # unfrozen leaf
        leaf_ = pytc.unfreeze(leaf)
        # array count is the product of the shape. if the node is not an array, then the count is 1
        count = int(np.array(leaf_.shape).prod()) if hasattr(leaf_, "shape") else 1
        size = leaf_.nbytes if hasattr(leaf_, "nbytes") else sys.getsizeof(leaf_)

        if pytc.is_frozen(tree) or pytc.is_frozen(leaf):
            count = complex(0, count)
            size = complex(0, size)

        counts += count
        sizes += size

    return (counts, sizes)


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

    max_width = (max(chain.from_iterable([[len(t) for t in item.split("\n")] for item in text])) + 0)  # fmt: skip

    top = f"┌{'─'*max_width}┐"
    line = f"├{'─'*max_width}┤"
    side = ["\n".join([f"│{t}{' '*(max_width-len(t))}│" for t in item.split("\n")]) for item in text]  # fmt: skip
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
    FMT = ""

    _cells = tuple(zip(*boxes))

    for i, line in enumerate(_cells):
        FMT += _resolve_line(line) + ("\n" if i != (len(_cells) - 1) else "")

    return FMT


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


def _table(lines: list[list[str]]) -> str:
    # Create a table with self aligning rows and cols

    # Args:
    #     lines (Sequence[str,...]): list of lists of cols values

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

    for i, _cells in enumerate(zip(*lines)):
        max_cell_height = max(map(lambda x: x.count("\n"), _cells))
        for j in range(len(_cells)):
            lines[j][i] += "\n" * (max_cell_height - lines[j][i].count("\n"))

    return _hstack(*(_vbox(*col) for col in lines))


def tree_summary(
    tree: PyTree, *, width: int = 60, depth: int | float = float("inf")
) -> str:
    """Print a summary of an arbitrary PyTree.

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
        >>> import pytreeclass as pytc
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
    ROWS = [["Name", "Type", "Count", "Size"]]

    traces, leaves = zip(
        *pytc.tree_leaves_with_trace(
            tree,
            is_leaf=pytc.is_frozen,
            is_trace_leaf=_is_trace_leaf_depth_factory(depth),
        )
    )
    # in case of a single node tree or depth=0, avoid printing the node twice
    # once for the trace and once for the summary
    traces = traces if len(traces) > 1 else ()

    for trace, leaf in zip(traces, leaves):
        names, types, _, metadatas = trace

        if _should_omit_trace(metadatas):
            continue

        row = [_resolve_names(names[1:], width)]

        # type name row
        row += [_node_type_pprint(pytc.unfreeze(leaf), 0, "str", width, depth)]

        # count and size row
        count, size = _calculate_leaf_trace_stats(leaf)
        leaves_count = _format_count(count.real + count.imag)
        leaves_size = _format_size(size.real + size.imag)

        # add frozen stats only if there are frozen leaves
        leaves_count += f"({_format_count(count.imag)})" if count.imag > 0 else ""
        leaves_size += f"({_format_size(size.imag)})" if size.imag > 0 else ""
        row += [leaves_count, leaves_size]

        ROWS += [row]

    COUNT = [complex(0), complex(0)]  # non-frozen, frozen
    SIZE = [complex(0), complex(0)]

    for trace, leaf in pytc.tree_leaves_with_trace(tree, is_leaf=pytc.is_frozen):
        count, size = _calculate_leaf_trace_stats(leaf)
        COUNT[pytc.is_frozen(leaf)] += count
        SIZE[pytc.is_frozen(leaf)] += size

    unfrozen_count = COUNT[0].real + COUNT[0].imag
    frozen_count = COUNT[1].real + COUNT[1].imag
    total_count = unfrozen_count + frozen_count

    unfrozen_size = SIZE[0].real + SIZE[0].imag
    frozen_size = SIZE[1].real + SIZE[1].imag
    total_size = unfrozen_size + frozen_size

    row = ["Σ"]
    row += [_node_type_pprint(tree, 0, "repr", width, depth)]
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


def tree_repr_with_trace(tree: PyTree) -> PyTree:
    """Return a PyTree with the same structure, but with the leaves replaced by a summary of the trace.

    Example:
        >>> import pytreeclass as pytc
        >>> @pytc.treeclass
        ... class Test:
        ...    a:int = 1
        ...    b:float = 2.0

        >>> tree = Test()
        >>> print(pytc.tree_repr_with_trace(Test()))
        Test(
        a=
            ┌──────────┬─────────┐
            │Value     │1        │
            ├──────────┼─────────┤
            │Name path │Test->a  │
            ├──────────┼─────────┤
            │Type path │Test->int│
            ├──────────┼─────────┤
            │Index path│0->0     │
            └──────────┴─────────┘,
        b=
            ┌──────────┬───────────┐
            │Value     │2.0        │
            ├──────────┼───────────┤
            │Name path │Test->b    │
            ├──────────┼───────────┤
            │Type path │Test->float│
            ├──────────┼───────────┤
            │Index path│0->1       │
            └──────────┴───────────┘
        )

    Note:
        This function can be useful for debugging and raising descriptive errors.
    """

    def leaf_trace_summary(trace, leaf) -> str:
        # this can be useful in debugging and raising descriptive errors
        ROWS = [["Value", tree_repr(leaf)]]

        names = "->".join(trace[0])
        ROWS += [["Name path", names]]

        types = "->".join(i.__name__ for i in trace[1])
        ROWS += [["Type path", types]]

        indices = "->".join(str(i) for i in trace[2])
        ROWS += [["Index path", indices]]

        # make a pretty table for each leaf
        COLS = [list(c) for c in zip(*ROWS)]

        return _table(COLS)

    return pytc.tree_map_with_trace(leaf_trace_summary, tree)
