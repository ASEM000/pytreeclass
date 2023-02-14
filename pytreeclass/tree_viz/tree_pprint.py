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
from pytreeclass.tree_viz.box_drawing import _table
from pytreeclass.tree_viz.tree_viz_util import (
    NodeInfo,
    _format_count,
    _format_size,
    _format_width,
    _marker,
    _mermaid_marker,
    tree_trace,
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

    if isinstance(node, (np.ndarray, jnp.ndarray)) and kind == "repr":
        # works for numpy arrays, jax arrays
        return _numpy_pprint(node, kind)

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

    if isinstance(node, jax.ShapeDtypeStruct):
        return _shape_dtype_struct_pprint(node)

    if kind not in ("repr", "str"):
        raise ValueError(f"kind must be 'repr' or 'str', got {kind}")

    fmt = f"{node!r}" if kind == "repr" else f"{node!s}"

    if "\n" not in fmt:
        # if no newlines, just return the string with the correct indentation
        return ("\n" + "\t" * (depth)).join(fmt.split("\n"))

    # if there are newlines, indent the string and return it with a newline
    return "\n" + "\t" * (depth + 1) + ("\n" + "\t" * (depth + 1)).join(fmt.split("\n"))


_printer_map = {
    "repr": lambda node, depth: _node_pprint(node, depth, kind="repr"),
    "str": lambda node, depth: _node_pprint(node, depth, kind="str"),
}


def _shape_dtype_struct_pprint(node: jax.ShapeDtypeStruct) -> str:
    """Pretty print jax.ShapeDtypeStruct"""
    shape = (
        f"{node.shape}".replace(",", "")
        .replace("(", "[")
        .replace(")", "]")
        .replace(" ", ",")
    )
    dtype = (
        f"{node.dtype}".replace("float", "f")
        .replace("int", "i")
        .replace("complex", "c")
    )
    return f"{dtype}{shape}"


def _numpy_pprint(node: np.ndarray, kind: str = "repr") -> str:
    """Replace np.ndarray repr with short hand notation for type and shape

    Example:
        >>> _numpy_pprint(np.ones((2,3)))
        'f64[2,3]∈[1.0,1.0]'
    """
    shape = (
        f"{node.shape}".replace(",", "")
        .replace("(", "[")
        .replace(")", "]")
        .replace(" ", ",")
    )
    if issubclass(node.dtype.type, np.integer):
        base = f"{node.dtype}".replace("int", "i") + shape
    elif issubclass(node.dtype.type, np.floating):
        base = f"{node.dtype}".replace("float", "f") + shape
    elif issubclass(node.dtype.type, np.complexfloating):
        base = f"{node.dtype}".replace("complex", "c") + shape
    else:
        base = f"{node.dtype}" + shape

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

    if kind == "repr":
        return base
    raise ValueError(f"kind must be 'repr' got {kind}")


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


def tree_diagram(tree: PyTree) -> str:
    """Pretty print treeclass tree with tree structure diagram
    Args:
        tree: treeclass tree (instance of dataclass)

    Example:
        >>> @pytc.treeclass
        ... class A:
        ...        x: int = 10
        ...        y: int = (20,30)

        >>> @pytc.treeclass
        ... class B:
        ...     a: int = 10
        ...     b: tuple = (20,30, A())  # will be expanded as it continas a dataclass in it
        ...     c: int = dc.field(default=40, repr=False) # will be skipped as repr=False
        ...     d: jnp.ndarray = dc.field(defaualt_factory=lambda: jnp.array([1,2,3]))
        ...     e: tuple = (1,2,3)

        >>> print(tree_diagram(B()))
        B
            ├── a:int=10
            ├── b:tuple
            │   ├── [0]:int=20
            │   ├── [1]:int=30
            │   └── [2]:A
            │       ├── x:int=10
            │       └── y:tuple=(20,30)
            ├── d:Array=i32[3]∈[1,3]
            └── e:tuple=(1,2,3)
    """

    def recurse(tree, path, index, expand):

        nonlocal fmt

        if not expand:
            return

        if dc.is_dataclass(tree):
            fields = [f for f in dc.fields(tree) if f.repr]
            count = len(fields)
            names = (f.name for f in fields)
            values = (getattr(tree, f.name) for f in fields)
            marks = (_marker(f, getattr(tree, f.name), default="─") for f in fields)

        elif isinstance(tree, dict):
            # expand dict if it contains dataclass
            names = tree.keys()
            values = tree.values()
            marks = ("-" for _ in tree)
            count = len(tree)

        else:
            # make it work for arbitrary PyTrees
            # however, no names will be shown
            values = tree if isinstance(tree, (list, tuple)) else jtu.tree_leaves(tree)
            count = len(values)
            names = (f"[{i}]" for i in range(count))
            marks = ("─" for _ in values)

        for i, (value, name, mark) in enumerate(zip(values, names, marks)):
            index = count - i
            is_last_field = index == 1

            fmt += "\n"
            fmt += "".join([(("│" if lvl > 1 else "") + "\t") for lvl in path])
            fmt += f"└{mark}─ " if is_last_field else f"├{mark}─ "
            fmt += f"{name}"

            # decide whether to expand a container or not
            expand = False

            if isinstance(value, (list, tuple, dict)):
                if len(f"{value}!r") > 60:  # expand if multiline expression
                    expand = True
                elif any(map(dc.is_dataclass, value)):
                    expand = True

            elif dc.is_dataclass(value):
                expand = True

            if expand:
                # expand dataclass or list/tuple with a single dataclass item
                # otherwise do not expand
                fmt += f":{value.__class__.__name__}"
                recurse(value, path + [index], index, expand=True)
                continue

            # leaf node case
            fmt += f":{value.__class__.__name__}={_node_pprint(value, kind='repr')}"
            recurse(value, path + [1], index, expand=False)

        fmt += "\t"

    fmt = f"{(tree.__class__.__name__)}"
    recurse(tree, [1], 0, expand=True)
    return fmt.expandtabs(4)


def _get_type_name(node):
    if isinstance(node, (jnp.ndarray, np.ndarray)):
        return _node_pprint(jax.ShapeDtypeStruct(node.shape, node.dtype))
    return f"{node.__class__.__name__}"


def tree_summary(
    tree: PyTree, *, depth=float("inf"), array: jnp.ndarray | None = None
) -> str:
    """Print a summary of a pytree structure

    Args:
        tree: pytree to summarize (ex. list, tuple, dict, dataclass, jax.numpy.ndarray)
        depth: depth to traverse the tree. defaults to maximum depth.

    Note:
        array elements are considered as leaves, for example `jnp.array([1,2,3])` has 3 leaves
    Example:
        >>> # Traverse only the first level of the tree
        >>> print(tree_summary((1,(2,(3,4))),depth=1))
    """
    ROWS = [["Name", "Type", "Count(Frozen)", "Size(Frozen)"]]

    # traverse the tree and collect info about each node
    infos = tree_trace(tree, depth)
    infos = infos if len(infos) > 1 else ()

    for info in infos:
        if not info.repr:
            continue

        path = ".".join(info.path).replace("].", "]").replace(".[", "[")

        row = [path]
        node = (info.node).unwrap() if pytc.is_frozen(info.node) else info.node

        # type name row
        row += [_get_type_name(node)]

        # count row
        frozen_count = info.count.imag
        total_count = info.count.real + frozen_count
        row += [f"{_format_count(total_count)}({_format_count(frozen_count)})"]

        # size row
        frozen_size = info.size.imag
        total_size = info.size.real + frozen_size
        row += [f"{_format_size(total_size)}({_format_size(frozen_size)})"]

        ROWS += [row]

    # add summary row at the end to
    # show total number of leaves and total size
    COUNT = [complex(0), complex(0)]  # non-frozen, frozen
    SIZE = [complex(0), complex(0)]

    for info in tree_trace(tree):
        COUNT[info.frozen] += info.count
        SIZE[info.frozen] += info.size

    unfrozen_count = COUNT[0].real + COUNT[0].imag
    frozen_count = COUNT[1].real + COUNT[1].imag
    total_count = unfrozen_count + frozen_count

    unfrozen_size = SIZE[0].real + SIZE[0].imag
    frozen_size = SIZE[1].real + SIZE[1].imag
    total_size = unfrozen_size + frozen_size

    row = ["Σ"]
    row += [_get_type_name(tree)]
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

        return ft.reduce(reduce_func, tree_trace(item), (0, 0))

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
