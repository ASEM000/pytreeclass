from __future__ import annotations

import math
import os
from dataclasses import field
from typing import Any

import jax.numpy as jnp

import pytreeclass as pytc
import pytreeclass._src as src
from pytreeclass._src.tree_base import _tree_structure

# from pytreeclass._src.dispatch import dispatch
from pytreeclass._src.tree_util import tree_unfreeze
from pytreeclass.tree_viz.box_drawing import _table
from pytreeclass.tree_viz.node_pprint import _format_node_repr
from pytreeclass.tree_viz.utils import (
    _reduce_count_and_size,
    _sequential_tree_shape_eval,
)

PyTree = Any


def _bold_text(text: str) -> str:
    return f"\033[1m{text}\033[0m"


def _format_size(node_size, newline=False):
    """return formatted size from inexact(exact) complex number"""
    mark = "\n" if newline else ""
    order_kw = ["B", "KB", "MB", "GB"]

    if isinstance(node_size, complex):
        # define order of magnitude
        real_size_order = (
            int(math.log(node_size.real, 1024)) if node_size.real > 0 else 0
        )
        imag_size_order = (
            int(math.log(node_size.imag, 1024)) if node_size.imag > 0 else 0
        )
        return (
            f"{(node_size.real)/(1024**real_size_order):.2f}{order_kw[real_size_order]}{mark}"
            f"({(node_size.imag)/(1024**imag_size_order):.2f}{order_kw[imag_size_order]})"
        )

    elif isinstance(node_size, (float, int)):
        size_order = int(math.log(node_size, 1024)) if node_size > 0 else 0
        return f"{(node_size)/(1024**size_order):.2f}{order_kw[size_order]}"


def _format_count(node_count, newline=False):
    mark = "\n" if newline else ""

    if isinstance(node_count, complex):
        return f"{int(node_count.real):,}{mark}({int(node_count.imag):,})"
    elif isinstance(node_count, (float, int)):
        return f"{int(node_count):,}"


def tree_summary(
    tree: PyTree,
    array: jnp.ndarray = None,
    *,
    show_type: bool = True,
    show_param: bool = True,
    show_size: bool = True,
    show_config: bool = True,
) -> str:
    """Prints a summary of the tree structure.

    Args:
        tree (PyTree): @treeclass decorated class.
        array (jnp.ndarray, optional): Input jax.numpy used to call the class. Defaults to None.
        show_type (bool, optional): Whether to print the type column. Defaults to True.
        show_param (bool, optional): Whether to print the parameter column. Defaults to True.
        show_size (bool, optional): Whether to print the size column. Defaults to True.
        show_config (bool, optional): Whether to print the config column. Defaults to True.

    Example:
        @pytc.treeclass
        class Test:
            a: int = 0
            b : jnp.ndarray = jnp.array([1,2,3])
            c : float = 1.0


        >>> print(tree_summary(Test()))
        ┌────┬─────┬───────┬─────────────┬──────┐
        │Name│Type │Param #│Size         │Config│
        ├────┼─────┼───────┼─────────────┼──────┤
        │a   │int  │0(1)   │0.00B(28.00B)│a=1   │
        ├────┼─────┼───────┼─────────────┼──────┤
        │b   │float│1(0)   │24.00B(0.00B)│b=2.0 │
        └────┴─────┴───────┴─────────────┴──────┘
        Total count :	1(1)
        Dynamic count :	1(1)
        Frozen count :	0(0)
        -----------------------------------------
        Total size :	24.00B(28.00B)
        Dynamic size :	24.00B(28.00B)
        Frozen size :	0.00B(0.00B)
        =========================================

    Note:
        values inside () defines the info about the non-inexact (i.e.) non-differentiable parameters.
        this distinction is important for the jax.grad function.
        to see which values types needs to be handled for training

    Returns:
        str: Summary of the tree structure.
    """
    _format_node = lambda node: _format_node_repr(node, depth=0).expandtabs(1)

    if array is not None:
        shape = _sequential_tree_shape_eval(tree, array)
        indim_shape, outdim_shape = shape[:-1], shape[1:]

        in_shape_str = ["Input"] + list(map(_format_node, indim_shape))
        out_shape_str = ["Output"] + list(map(_format_node, outdim_shape))

    def recurse_field(field_item, node_item, name_path, type_path):

        nonlocal ROWS, COUNT, SIZE

        if not field_item.repr:
            return

        if isinstance(node_item, (list, tuple)) and any(
            pytc.is_treeclass(leaf) for leaf in node_item
        ):
            # expand container if any item is a `treeclass`
            for i, layer in enumerate(node_item):
                new_field = field(metadata={"frozen": pytc.is_frozen_field(field_item)})
                object.__setattr__(new_field, "name", f"{field_item.name}[{i}]")
                object.__setattr__(new_field, "type", type(layer))

                recurse_field(
                    field_item=new_field,
                    node_item=layer,
                    name_path=name_path + (f"{field_item.name}[{i}]",),
                    type_path=type_path + (layer.__class__.__name__,),
                )

        elif isinstance(node_item, src.tree_base._treeBase):
            # a module is considred frozen if all it's parameters are frozen
            pytc.is_frozen = field_item.metadata.get("frozen", False)
            pytc.is_frozen = pytc.is_frozen or pytc.is_treeclass_frozen(node_item)
            count, size = _reduce_count_and_size(tree_unfreeze(node_item))
            dynamic, _ = _tree_structure(tree_unfreeze(node_item))

            row = ["/".join(name_path) + f"{(os.linesep + '(frozen)' if pytc.is_frozen else '')}"]  # fmt: skip
            row += ["/".join(type_path)] if show_type else []
            row += [_format_count(count)] if show_param else []
            row += [_format_size(size)] if show_size else []
            row += (
                ["\n".join([f"{k}={_format_node(v)}" for k, v in dynamic.items()])]
                if show_config
                else []
            )

            ROWS.append(row)

            COUNT[1 if pytc.is_frozen else 0] += count
            SIZE[1 if pytc.is_frozen else 0] += size

        else:
            pytc.is_frozen = field_item.metadata.get("frozen", False)
            count, size = _reduce_count_and_size(node_item)

            row = ["/".join(name_path) + f"{(os.linesep + '(frozen)' if pytc.is_frozen else '')}"]  # fmt: skip
            row += ["/".join(type_path)] if show_type else []
            row += [_format_count(count)] if show_param else []
            row += [_format_size(size)] if show_size else []
            row += [f"{field_item.name}={_format_node(node_item)}"] if show_config else []  # fmt: skip

            ROWS.append(row)

            # non-treeclass leaf inherit frozen state
            COUNT[1 if pytc.is_frozen else 0] += count
            SIZE[1 if pytc.is_frozen else 0] += size

    def recurse(tree, name_path, type_path):

        nonlocal ROWS, COUNT, SIZE

        for field_item in pytc.fields(tree):

            node_item = getattr(tree, field_item.name)

            if pytc.is_treeclass_non_leaf(node_item):
                # recurse if the field is a treeclass
                # the recursion passes the frozen state of the current node
                # name_path,type_path (i.e. location of the ndoe in the tree)
                # for instance a path "L1/L0" defines a class L0 with L1 parent
                recurse(
                    tree=node_item,
                    name_path=name_path + (field_item.name,),
                    type_path=type_path + (node_item.__class__.__name__,),
                )

            else:

                pytc.is_static = field_item.metadata.get("static", False)
                pytc.is_static = pytc.is_static and not field_item.metadata.get(
                    "frozen", False
                )

                # skip if the field is static and not frozen
                if not (pytc.is_static):
                    recurse_field(
                        field_item=field_item,
                        node_item=node_item,
                        name_path=name_path + (field_item.name,),
                        type_path=type_path + (node_item.__class__.__name__,),
                    )

    row = ["Name"]
    row += ["Type "] if show_type else []
    row += ["Param #"] if show_param else []
    row += ["Size "] if show_size else []
    row += ["Config"] if show_config else []

    ROWS = [row]

    COUNT = [complex(0), complex(0)]
    SIZE = [complex(0), complex(0)]

    recurse(tree, name_path=(), type_path=())

    # we need to transform rows to cols
    # as `_table` concatenates columns together
    COLS = [list(c) for c in zip(*ROWS)]

    if array is not None:
        COLS += [in_shape_str]
        COLS += [out_shape_str]

    layer_table = _table(COLS)
    table_width = len(layer_table.split("\n")[0])

    param_summary = (
        f"Total count :\t{_format_count(sum(COUNT))}\n"
        f"Dynamic count :\t{_format_count(COUNT[0])}\n"
        f"Frozen count :\t{_format_count(COUNT[1])}\n"
        f"{'-'*max([table_width,40])}\n"
        f"Total size :\t{_format_size(sum(SIZE))}\n"
        f"Dynamic size :\t{_format_size(SIZE[0])}\n"
        f"Frozen size :\t{_format_size(SIZE[1])}\n"
        f"{'='*max([table_width,40])}"
    )

    return layer_table + "\n" + (param_summary)
