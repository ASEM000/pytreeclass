from __future__ import annotations

import ctypes
import dataclasses as dc
import functools as ft
from typing import Any

import jax.tree_util as jtu

from pytreeclass.tree_viz.box_drawing import _table
from pytreeclass.tree_viz.node_pprint import _node_pprint
from pytreeclass.tree_viz.tree_viz_util import (
    NodeInfo,
    _format_count,
    _format_size,
    _format_width,
    _marker,
    _mermaid_marker,
    tree_trace,
)

""" Pretty print pytress"""

PyTree = Any

__all__ = ("tree_diagram", "tree_repr", "tree_str", "tree_summary")


def _tree_pprint(tree, width: int = 80, kind="repr") -> str:
    """Prertty print dataclass tree"""

    def recurse(tree: PyTree, depth: int):
        if not dc.is_dataclass(tree):
            return

        nonlocal FMT

        leaves_count = len(dc.fields(tree))
        fields = [f for f in dc.fields(tree) if f.repr]
        leaves_count = len(fields)

        for i, field_item in enumerate(fields):
            node = getattr(tree, field_item.name)
            mark = _marker(field_item, node)
            endl = "" if i == (leaves_count - 1) else ", "
            FMT += "\n" + "\t" * depth

            if dc.is_dataclass(node):
                FMT += f"{mark}{field_item.name}={node.__class__.__name__}("
                cursor = len(FMT)  # capture children repr
                recurse(tree=node, depth=depth + 1)
                _FMT = FMT
                FMT = _FMT[:cursor]
                FMT += _format_width(_FMT[cursor:] + "\n" + "\t" * (depth) + ")")
                FMT += endl
                continue

            # leaf node case
            FMT += f"{mark}{field_item.name}="

            if kind == "repr":
                FMT += f"{(_node_pprint(node,depth, kind=kind))}"

            elif kind == "str":
                if "\n" in f"{node!s}":
                    # in case of multiline string then indent all lines
                    FMT += "\n" + "\t" * (depth + 1)
                    FMT += f"{(_node_pprint(node,depth+1, kind=kind))}"
                else:
                    FMT += f"{(_node_pprint(node,depth, kind=kind))}"

            FMT += endl
            recurse(tree=node, depth=depth)

    FMT = ""
    recurse(tree=tree, depth=1)
    FMT = f"{(tree.__class__.__name__)}(" + _format_width(FMT + "\n)", width)
    return FMT.expandtabs(2)


def tree_repr(tree, width: int = 80) -> str:
    """Prertty print dataclass tree __repr__"""
    return _tree_pprint(tree, width, kind="repr")


def tree_str(tree, width: int = 80) -> str:
    """Prertty print dataclass tree __str__"""
    return _tree_pprint(tree, width, kind="str")


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

        nonlocal FMT

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

            FMT += "\n"
            FMT += "".join([(("│" if lvl > 1 else "") + "\t") for lvl in path])
            FMT += f"└{mark}─ " if is_last_field else f"├{mark}─ "
            FMT += f"{name}"

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
                FMT += f":{value.__class__.__name__}"
                recurse(value, path + [index], index, expand=True)
                continue

            # leaf node case
            FMT += f":{value.__class__.__name__}={_node_pprint(value, kind='repr')}"
            recurse(value, path + [1], index, expand=False)

        FMT += "\t"

    FMT = f"{(tree.__class__.__name__)}"
    recurse(tree, [1], 0, expand=True)
    return FMT.expandtabs(4)


def tree_summary(tree: PyTree, *, depth=float("inf")) -> str:
    """Print a summary of the pytree structure

    Args:
        tree: pytree to summarize (ex. list, tuple, dict, dataclass, jax.numpy.ndarray)
        depth: depth to traverse the tree (default: float("inf"))

    Note:
        array elements are considered as leaves, for example `jnp.array([1,2,3])` has 3 leaves
    Example:
        >>> # Traverse only the first level of the tree
        >>> print(tree_summary((1,(2,(3,4))),depth=1))
        ┌────┬─────┬──────┬──────┬─────────────┐
        │Name│Type │Leaf #│Size  │Config       │
        ├────┼─────┼──────┼──────┼─────────────┤
        │[0] │int  │1     │28.00B│[0]=1        │
        ├────┼─────┼──────┼──────┼─────────────┤
        │[1] │tuple│3     │84.00B│[1]=(2,(3,4))│
        └────┴─────┴──────┴──────┴─────────────┘
        Total leaf count:       4
        Non-frozen leaf count:  4
        Frozen leaf count:      0
        ----------------------------------------
        Total leaf size:        112.00B
        Non-frozen leaf size:   112.00B
        Frozen leaf size:       0.00B
        ========================================

        >>> # Traverse two levels of the tree
        >>> print(tree_summary((1,(2,(3,4))),depth=2))
        ┌──────┬─────┬──────┬──────┬────────────┐
        │Name  │Type │Leaf #│Size  │Config      │
        ├──────┼─────┼──────┼──────┼────────────┤
        │[0]   │int  │1     │28.00B│[0]=1       │
        ├──────┼─────┼──────┼──────┼────────────┤
        │[1][0]│int  │1     │28.00B│[1][0]=2    │
        ├──────┼─────┼──────┼──────┼────────────┤
        │[1][1]│tuple│2     │56.00B│[1][1]=(3,4)│
        └──────┴─────┴──────┴──────┴────────────┘
        Total leaf count:       4
        Non-frozen leaf count:  4
        Frozen leaf count:      0
        -----------------------------------------
        Total leaf size:        112.00B
        Non-frozen leaf size:   112.00B
        Frozen leaf size:       0.00B
        =========================================

    """
    ROWS = [["Name", "Type ", "Leaf #", "Size ", "Config"]]
    COUNT = [complex(0), complex(0)]  # non-frozen, frozen
    SIZE = [complex(0), complex(0)]

    for info in tree_trace(tree, depth):
        # `tree_trace` returns a list of `NodeInfo` objects that contain the
        # leaves info at the specified depth
        row = [info.path]  # name
        row += [f"{info.node.__class__.__name__}" + ("(Frozen)" if info.frozen else "")]
        row += [_format_count(info.count.real + info.count.imag)]
        row += [_format_size(info.size.real + info.size.imag)]
        row += [
            f"{info.path.split('.')[-1]}={_node_pprint(info.node, kind='repr').expandtabs(1)}"
        ]
        # row += [str(info.frozen)]  # frozen
        ROWS += [row]
        COUNT[int(info.frozen)] += info.count
        SIZE[int(info.frozen)] += info.size

    COLS = [list(c) for c in zip(*ROWS)]
    layer_table = _table(COLS)
    table_width = len(layer_table.split("\n")[0])

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
        nonlocal FMT

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

                FMT += f"\n\tid{prev_id} {mark} "
                FMT += f'|"{(count)}<br>{(size)}"| '  # add count and size of children
                FMT += f'id{cur_id}("{bold_text(name) }:{type}")'
                recurse(tree=value, depth=depth + 1, prev_id=cur_id, expand=True)
                continue

            FMT += f"\n\tid{prev_id} {mark} "

            # add count and size of children
            if mark == "-..-":
                FMT += f'|"(frozen) {(count)}<br>{(size)}"| '  # add count and size of children for frozen leaves
            elif mark == "--x":
                FMT += ""  # no count or size for frozen leaves
            elif mark == "--->" or mark == "---":
                FMT += f'|"{(count)}<br>{(size)}"| '

            FMT += f'id{cur_id}["{bold_text(name)}:{type}={_node_pprint(value, kind="repr")}"]'

        prev_id = cur_id

    cur_id = node_id((0, 0, -1, 0))
    FMT = f"flowchart LR\n\tid{cur_id}({bold_text(tree.__class__.__name__)})"
    recurse(tree=tree, depth=1, prev_id=cur_id)
    return FMT.expandtabs(4)
