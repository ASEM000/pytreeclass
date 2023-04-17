from __future__ import annotations

import dataclasses as dc
import functools as ft
import inspect
import math
from collections import defaultdict
from itertools import chain
from types import FunctionType
from typing import Any, Callable, Literal

import jax
import numpy as np
from jax._src.custom_derivatives import custom_jvp
from jax.util import unzip2
from jaxlib.xla_extension import PjitFunction

import pytreeclass as pytc

PyTree = Any
PrintKind = Literal["repr", "str"]


def _node_pprint(
    node: Any,
    indent: int,
    kind: PrintKind,
    width: int,
    depth: int | float,
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
    if isinstance(node, pytc.TreeClass):
        return _treeclass_pprint(node, indent, kind, width, depth)
    return _general_pprint(node, indent, kind, width, depth)


def _general_pprint(
    node: Any,
    indent: int,
    kind: PrintKind,
    width: int,
    depth: int,
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
    dtype = f"{node.dtype}".replace("int", "i")
    dtype = dtype.replace("float", "f")
    dtype = dtype.replace("complex", "c")
    return _format_width(dtype + shape, width)


def _numpy_pprint(
    node: np.ndarray | jax.Array,
    indent: int,
    kind: PrintKind,
    width: int,
    depth: int,
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
    func: Callable,
    indent: int,
    kind: PrintKind,
    width: int,
    depth: int,
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
    node: list,
    indent: int,
    kind: PrintKind,
    width: int,
    depth: int,
) -> str:
    if depth == 0:
        fmt = "..."
    else:
        fmt = (f"{(_node_pprint(v,indent+1,kind,width,depth-1))}" for v in node)
        fmt = (", \n" + "\t" * (indent + 1)).join(fmt)

    fmt = "[\n" + "\t" * (indent + 1) + (fmt) + "\n" + "\t" * (indent) + "]"
    return _format_width(fmt, width)


def _tuple_pprint(
    node: tuple,
    indent: int,
    kind: PrintKind,
    width: int,
    depth: int,
) -> str:
    if depth == 0:
        fmt = "..."
    else:
        fmt = (f"{(_node_pprint(v,indent+1,kind,width,depth-1))}" for v in node)
        fmt = (", \n" + "\t" * (indent + 1)).join(fmt)
    fmt = "(\n" + "\t" * (indent + 1) + (fmt) + "\n" + "\t" * (indent) + ")"
    return _format_width(fmt, width)


def _set_pprint(
    node: set,
    indent: int,
    kind: PrintKind,
    width: int,
    depth: int,
) -> str:
    if depth == 0:
        fmt = "..."
    else:
        fmt = (f"{(_node_pprint(v,indent+1,kind,width,depth-1))}" for v in node)
        fmt = (", \n" + "\t" * (indent + 1)).join(fmt)
    fmt = "{\n" + "\t" * (indent + 1) + (fmt) + "\n" + "\t" * (indent) + "}"
    return _format_width(fmt, width)


def _dict_pprint(
    node: dict,
    indent: int,
    kind: PrintKind,
    width: int,
    depth: int,
) -> str:
    if depth == 0:
        fmt = "..."
    else:
        kvs = node.items()
        fmt = (f"{k}:{_node_pprint(v,indent+1,kind,width,depth-1)}" for k, v in kvs)
        fmt = (", \n" + "\t" * (indent + 1)).join(fmt)
    fmt = "{\n" + "\t" * (indent + 1) + (fmt) + "\n" + "\t" * (indent) + "}"
    return _format_width(fmt, width)


def _namedtuple_pprint(
    node: Any,
    indent: int,
    kind: PrintKind,
    width: int,
    depth: int,
) -> str:
    if depth == 0:
        fmt = "..."
    else:
        kvs = node._asdict().items()
        fmt = (f"{k}={_node_pprint(v,indent+1,kind,width,depth-1)}" for k, v in kvs)
        fmt = (", \n" + "\t" * (indent + 1)).join(fmt)
    name = type(node).__name__
    fmt = f"{name}(\n" + "\t" * (indent + 1) + (fmt) + "\n" + "\t" * (indent) + ")"
    return _format_width(fmt, width)


def _dataclass_pprint(
    node: Any,
    indent: int,
    kind: PrintKind,
    width: int,
    depth: int,
) -> str:
    name = type(node).__name__

    if depth == 0:
        fmt = "..."
    else:
        kvs = ((f.name, vars(node)[f.name]) for f in dc.fields(node) if f.repr)
        fmt = (f"{k}={_node_pprint(v,indent+1,kind,width,depth-1)}" for k, v in kvs)
        fmt = (", \n" + "\t" * (indent + 1)).join(fmt)

    fmt = f"{name}(\n" + "\t" * (indent + 1) + (fmt) + "\n" + "\t" * (indent) + ")"
    return _format_width(fmt, width)


def _treeclass_pprint(
    node: Any,
    indent: int,
    kind: PrintKind,
    width: int,
    depth: int,
) -> str:
    name = type(node).__name__
    if depth == 0:
        fmt = "..."

    else:
        kvs = ((k, vars(node)[k]) for k, f in node._fields.items() if f.repr)
        fmt = (f"{k}={_node_pprint(v,indent+1,kind,width,depth-1)}" for k, v in kvs)
        fmt = (", \n" + "\t" * (indent + 1)).join(fmt)
    fmt = f"{name}(\n" + "\t" * (indent + 1) + (fmt) + "\n" + "\t" * (indent) + ")"
    return _format_width(fmt, width)


def _node_type_pprint(
    node: jax.Array | np.ndarray,
    indent: int,
    kind: PrintKind,
    width: int,
    depth: int,
) -> str:
    if isinstance(node, (jax.Array, np.ndarray)):
        shape_dype = node.shape, node.dtype
        fmt = _node_pprint(jax.ShapeDtypeStruct(*shape_dype), indent, kind, width, depth)  # fmt: skip
    else:
        fmt = f"{type(node).__name__}"
    return _format_width(fmt, width)


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
        {...}

        >>> print(pytc.tree_repr(tree, depth=1))
        {a:1, b:[...], c:{...}, f:i32[2](μ=6.50, σ=0.50, ∈[6,7])}

        >>> print(pytc.tree_repr(tree, depth=2))
        {a:1, b:[2, 3], c:{d:4, e:5}, f:i32[2](μ=6.50, σ=0.50, ∈[6,7])}
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

        >>> print(pytc.tree_str(tree, depth=1))
        {a:1, b:[...], c:{...}, f:[6 7]}

        >>> print(pytc.tree_str(tree, depth=2))
        {a:1, b:[2, 3], c:{d:4, e:5}, f:[6 7]}
    """
    return _node_pprint(tree, 0, "str", width, depth).expandtabs(tabwidth)


def _resolve_names(trace, width: int) -> str:
    # given a trace with a tuple of names, we resolve the names
    # to a single string
    names, _, indices = trace
    path = ""
    for i, (name, index) in enumerate(zip(names, indices)):
        name = f"[{index}]" if name is None else name
        path += "" if name.startswith("[") else ("." if i > 0 else "")
        path += _node_pprint(name, 0, "str", width, float("inf"))
    return path


def _is_trace_leaf_depth_factory(depth: int):
    # generate `is_trace_leaf` function to stop tracing at a certain `depth`
    # in essence, depth is the length of the trace entry
    def is_trace_leaf(trace) -> bool:
        # trace is a tuple of (names, leaves, tracers, aux_data)
        # done like this to ensure 4-tuple unpacking
        names, _, __ = trace
        # stop tracing if depth is reached
        return False if depth is None else (depth <= len(names))

    return is_trace_leaf


def tree_indent(
    tree: Any,
    *,
    width: int = 60,
    depth: int | float = float("inf"),
    is_leaf: Callable[[Any], bool] | None = None,
    tabwidth: int | None = 4,
):
    """Returns a string representation of the tree with indentation.

    Args:
        tree: The tree to be printed.
        width: The maximum width of leaf nodes line.
        depth: The maximum depth of the tree to be printed.
        is_leaf: A function that takes a node and returns True if it is a leaf node.
        tabwidth: The number of spaces per indentation level. if `None` then tabs are not expanded.

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
    fmt = f"{type(tree).__name__}"
    seen = set()

    for trace, leaf in pytc.tree_leaves_with_trace(
        tree=tree,
        is_leaf=is_leaf,
        is_trace_leaf=_is_trace_leaf_depth_factory(depth),
    ):
        names, types, indices = trace

        for j, (name, type_, index) in enumerate(zip(names, types, indices)):
            if (cur := (names[: j + 1], types[: j + 1], indices[: j + 1])) in seen:
                # skip printing the common parent node twice
                continue
            seen.add(cur)

            name = f"[{index}]" if name is None else name
            fmt += "\n" + "\t" * (j + 1)
            fmt += f"{_node_pprint(name,0,'str',width, depth )}"
            fmt += (
                f"={_node_pprint(leaf,indent=j+1,kind='repr',width=width, depth=depth-1)}"
                if j == len(names) - 1
                else f":{type_.__name__}"
            )
    return fmt if tabwidth is None else fmt.expandtabs(tabwidth)


def _group_by_depth(input: str) -> dict[int, list[list[int]]]:
    # >>> out = """L2
    # 	e=4
    # 	f:L1
    # 		c:L0
    # 			a=1
    # 			b=2
    # 		d=3
    # 	g:L0
    # 		a=1
    # 		b=2
    # 	h=5"""
    #
    # >>> print(_group_by_depth(out))
    # {
    #   3:[[4, 5]],
    #   2:[[3, 6], [8, 9]],
    #   0:[[0]],
    #   1:[[1, 2, 7, 10]]
    # }
    # in essence, the map key is the depth of the node, and the value is a list of line indices
    # each list of line indices is a parent node with line indices of its children
    depth_line_index_map = defaultdict(list)
    stack_map = defaultdict(list)
    prev_depth = 0
    lines = input.splitlines()

    for line_index, line in enumerate(lines):
        cur_depth = len(line) - len(line.lstrip("\t"))
        stack_map[cur_depth] += [line_index]
        if cur_depth < prev_depth and prev_depth in stack_map:
            depth_line_index_map[prev_depth] += [stack_map.pop(prev_depth)]
        prev_depth = cur_depth

    for key in stack_map:
        depth_line_index_map[key] += [stack_map[key]]
    del stack_map

    return dict(depth_line_index_map)


def _indent_to_diagram(input: str, tabwidth: int = 4) -> str:
    # input is a string of tab \t indented text
    # conversion alphabet
    vmark = ("│\t")[:tabwidth]  # vertical mark
    lmark = ("└" + "─" * (tabwidth - 2) + (" \t"))[:tabwidth]  # last mark
    cmark = ("├" + "─" * (tabwidth - 2) + (" \t"))[:tabwidth]  # connector mark
    smark = (" \t")[:tabwidth]  # space mark

    depth_line_index_map = _group_by_depth(input)
    lines = input.splitlines()

    for depth in depth_line_index_map:
        # iterate over groups of line indices at this depth
        for parent_lines_indices in depth_line_index_map[depth]:
            # iterate over line indices groups
            for i, line_index in enumerate(parent_lines_indices):
                # iterate over line indices at a group at this depth
                is_last = i == len(parent_lines_indices) - 1

                marker = ""
                for j in range(1, depth):
                    max_line_index = depth_line_index_map.get(j, [[-1]])[-1][-1]
                    marker += vmark if max_line_index > line_index else smark

                if depth > 0:
                    marker += lmark if is_last else cmark

                lines[line_index] = marker + lines[line_index].lstrip("\t")

    return "\n".join(lines).expandtabs(tabwidth)


def tree_diagram(
    tree: Any,
    *,
    width: int = 60,
    depth: int | float = float("inf"),
    is_leaf: Callable[[Any], bool] | None = None,
    tabwidth: int = 4,
):
    """Pretty print arbitrary PyTrees tree with tree structure diagram.

    Args:
        tree: PyTree
        depth: depth of the tree to print. default is max depth
        width: max width of line. default is 60
        is_leaf: function to determine if a node is a leaf. default is None

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
        B

        >>> print(pytc.tree_diagram(B(), depth=1))
        B
        ├── a=10
        └── b=(...)


        >>> print(pytc.tree_diagram(B(), depth=2))
        B
        ├── a=10
        └── b:tuple
            ├── [0]=20
            ├── [1]=30
            └── [2]=A(x=10, y=(...), z=40)
    """
    indent_repr = tree_indent(
        tree,
        width=width,
        depth=depth,
        is_leaf=is_leaf,
        tabwidth=None,
    )

    return _indent_to_diagram(indent_repr, tabwidth=tabwidth)


def _indent_to_mermaid(input: str, tabwidth: int) -> str:
    # input is a string of tab \t indented text

    depth_line_index_map = _group_by_depth(input)
    lines = input.splitlines()

    output = "flowchart LR\n"

    for depth in depth_line_index_map:
        # iterate over groups of line indices at this depth
        for parent_lines_indices in depth_line_index_map[depth]:
            # iterate over line indices groups
            for line_index in parent_lines_indices:
                if depth == 0:
                    line = "<b>" + lines[line_index].lstrip("\t") + "</b>"
                    output += f"\tid{line_index}({line})\n"

                else:
                    # get the line indices of the previous depth (parent nodes)
                    parent_lines = chain.from_iterable(depth_line_index_map[depth - 1])
                    parent_line_index = [x for x in parent_lines if x < line_index][-1]
                    line = "</b>" + lines[line_index].lstrip("\t") + "</b>"
                    output += f'\tid{parent_line_index} --- id{line_index}("{line}")\n'
    return output.expandtabs(tabwidth)


def tree_mermaid(
    tree: PyTree,
    width: int = 60,
    depth: int | float = float("inf"),
    is_leaf: Callable[[Any], bool] | None = None,
) -> str:
    # def _generate_mermaid_link(mermaid_string: str) -> str:
    #     """generate a one-time link mermaid diagram"""
    #     url_val = "https://pytreeclass.herokuapp.com/generateTemp"
    #     request = requests.post(url_val, json={"description": mermaid_string})
    #     generated_id = request.json()["id"]
    #     generated_html = f"https://pytreeclass.herokuapp.com/temp/?id={generated_id}"
    #     return f"Open URL in browser: {generated_html}"

    """generate a mermaid diagram syntax for arbitrary PyTrees.


    Args:
        tree: PyTree
        width: max width of line. default is 60
        depth: depth of the tree to print. default is max depth
        is_leaf: function to determine if a node is a leaf. default is None

    Example:
        >>> import pytreeclass as pytc
        >>> tree = [1, [2, 3], [4, [5, 6]]]
        >>> print(pytc.tree_mermaid(tree, depth=1))  # doctest: +SKIP
        flowchart LR
            id2 --- id3("</b>[0]=2</b>")
            id2 --- id4("</b>[1]=3</b>")
            id5 --- id6("</b>[0]=4</b>")
            id5 --- id7("</b>[1]:list</b>")
            id0(<b>list</b>)
            id0 --- id1("</b>[0]=1</b>")
            id0 --- id2("</b>[1]:list</b>")
            id0 --- id5("</b>[2]:list</b>")
            id7 --- id8("</b>[0]=5</b>")
            id7 --- id9("</b>[1]=6</b>")
    """

    indent_repr = tree_indent(
        tree,
        width=width,
        depth=depth,
        is_leaf=is_leaf,
        tabwidth=None,
    )

    return _indent_to_mermaid(indent_repr, tabwidth=4)


def _format_width(string, width=60):
    """strip newline/tab characters if less than max width"""
    children_length = len(string) - string.count("\n") - string.count("\t")
    if children_length > width:
        return string
    return string.replace("\n", "").replace("\t", "")


def _calculate_count(leaf: Any) -> tuple[int, int]:
    if hasattr(leaf, "shape") and hasattr(leaf, "nbytes"):
        return int(np.array(leaf.shape).prod())
    return 1


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


def tree_summary(
    tree: PyTree,
    *,
    depth: int | float = float("inf"),
    is_leaf: Callable[[Any], bool] | None = None,
) -> str:
    """Print a summary of an arbitrary PyTree.

    Args:
        tree: pytree to summarize (ex. list, tuple, dict, dataclass, jax.numpy.ndarray)
        depth: max depth to traverse the tree. defaults to maximum depth = float("inf")
        is_leaf: function to determine if a node is a leaf. defaults to None

    Returns:
        String summary of the tree structure
        - First column: is the path to the node
        - Second column: is the type of the node
        - Third column: is the number of leaves in the node (1 for non-array leaves and array size for array leaves)
        - Last row: type of parent, number of leaves and size of parent

    Note:
        Array elements are considered as leaves, for example `jnp.array([1,2,3])` has 3 leaves

    Example:
        >>> import pytreeclass as pytc
        >>> print(pytc.tree_summary([1,[2,[3]]]))
        ┌─────────┬────┬─────┐
        │Name     │Type│Count│
        ├─────────┼────┼─────┤
        │[0]      │int │1    │
        ├─────────┼────┼─────┤
        │[1][0]   │int │1    │
        ├─────────┼────┼─────┤
        │[1][1][0]│int │1    │
        ├─────────┼────┼─────┤
        │Σ        │list│3    │
        └─────────┴────┴─────┘
    """
    ROWS = [["Name", "Type", "Count"]]
    COUNT = 0

    # use `unzip2` from `jax.util` to avoid [] leaves
    # based on this issue:
    traces, leaves = unzip2(
        pytc.tree_leaves_with_trace(
            tree,
            is_leaf=is_leaf,
            is_trace_leaf=_is_trace_leaf_depth_factory(depth),
        )
    )

    for trace, leaf in zip(traces, leaves):
        count = _calculate_count(leaf)
        COUNT += count

        if trace == ((), (), ()):
            # avoid printing the leaf trace (which is the root of the tree)
            # twice, once as a leaf and once as the root at the end
            continue

        paths = _resolve_names(trace, width=60)
        types = _node_type_pprint(leaf, indent=0, kind="str", width=60, depth=depth)
        counts = f"{count:,}"
        ROWS += [[paths, types, counts]]

    paths = "Σ"
    types = _node_type_pprint(tree, indent=0, kind="str", width=60, depth=depth)
    counts = f"{COUNT:,}"
    ROWS += [[paths, types, counts]]

    return _table(ROWS)


def tree_repr_with_trace(
    tree: PyTree,
    is_leaf: Callable[[Any], bool] | None = None,
    transpose: bool = False,
) -> PyTree:
    """Return a PyTree with the same structure, but with the leaves replaced by a summary of the trace.

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
        >>> print(pytc.tree_repr_with_trace(Test()))  # doctest: +SKIP
        Test(
          a=
            ┌──────────┬───┐
            │Value     │1  │
            ├──────────┼───┤
            │Name path │a  │
            ├──────────┼───┤
            │Type path │int│
            ├──────────┼───┤
            │Index path│0  │
            └──────────┴───┘,
          b=
            ┌──────────┬─────┐
            │Value     │2.0  │
            ├──────────┼─────┤
            │Name path │b    │
            ├──────────┼─────┤
            │Type path │float│
            ├──────────┼─────┤
            │Index path│1    │
            └──────────┴─────┘
        )

        >>> print(pytc.tree_repr_with_trace(Test(), transpose=True))  # doctest: +SKIP
        Test(
        a=
            ┌─────┬─────────┬─────────┬──────────┐
            │Value│Name path│Type path│Index path│
            ├─────┼─────────┼─────────┼──────────┤
            │1    │a        │int      │0         │
            └─────┴─────────┴─────────┴──────────┘,
        b=
            ┌─────┬─────────┬─────────┬──────────┐
            │Value│Name path│Type path│Index path│
            ├─────┼─────────┼─────────┼──────────┤
            │2.0  │b        │float    │1         │
            └─────┴─────────┴─────────┴──────────┘
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
        return _table(ROWS, transpose=transpose)

    return pytc.tree_map_with_trace(leaf_trace_summary, tree, is_leaf=is_leaf)
