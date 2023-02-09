from __future__ import annotations

import ctypes
import dataclasses as dc
import functools as ft
from typing import Any

import jax.tree_util as jtu

import pytreeclass as pytc
from pytreeclass.tree_viz.box_drawing import _table
from pytreeclass.tree_viz.node_pprint import _node_pprint
from pytreeclass.tree_viz.tree_viz_util import (
    NodeInfo,
    _format_count,
    _format_size,
    _marker,
    _mermaid_marker,
    tree_trace,
)

""" Pretty print pytress"""

PyTree = Any

__all__ = ("tree_diagram", "tree_repr", "tree_str", "tree_summary")


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


def tree_summary(tree: PyTree, *, depth=float("inf")) -> str:
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
    ROWS = [["Name", "Type ", "Leaf #(size)", "Frozen #(size)", "Type stats"]]

    for info in tree_trace(tree, depth):
        path = ".".join(info.path).replace("].", "]").replace(".[", "[")

        row = [path]
        node = (info.node).unwrap() if pytc.is_frozen(info.node) else info.node
        row += [f"{node.__class__.__name__}"]

        # size and count
        leaves_count = _format_count(info.count.real)
        frozen_count = _format_count(info.count.imag)

        leaves_size = _format_size(info.size.real)
        frozen_size = _format_size(info.size.imag)

        row += [f"{leaves_count}({leaves_size})"]
        row += [f"{frozen_count}({frozen_size})"]

        # type stats
        row += [", ".join([f"{k}:{v:,}" for k, v in info.stats.items()])]

        ROWS += [row]

    COLS = [list(c) for c in zip(*ROWS)]
    layer_table = _table(COLS)
    table_width = len(layer_table.split("\n")[0])

    COUNT = [complex(0), complex(0)]  # non-frozen, frozen
    SIZE = [complex(0), complex(0)]

    for info in tree_trace(tree):
        COUNT[info.frozen] += info.count
        SIZE[info.frozen] += info.size

    total_count = sum(COUNT).real + sum(COUNT).imag
    non_frozen_count = COUNT[0].real + COUNT[0].imag
    frozen_count = total_count - non_frozen_count

    total_size = sum(SIZE).real + sum(SIZE).imag
    non_frozen_size = SIZE[0].real + SIZE[0].imag
    frozen_size = total_size - non_frozen_size

    param_summary = (
        f"Total leaf count:\t{_format_count(total_count)}\n"
        f"Non-frozen leaf count:\t{_format_count(non_frozen_count)}\n"
        f"Frozen leaf count:\t{_format_count(frozen_count)}\n"
        f"{'-'*max([table_width,40])}\n"
        f"Total leaf size:\t{_format_size(total_size)}\n"
        f"Non-frozen leaf size:\t{_format_size(non_frozen_size)}\n"
        f"Frozen leaf size:\t{_format_size(frozen_size)}\n"
        f"{'='*max([table_width,40])}\n"
    )

    return (layer_table + "\n" + (param_summary)).expandtabs(8)


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
