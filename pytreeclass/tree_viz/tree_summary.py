from __future__ import annotations

import dataclasses as dc
import math
from typing import Any

import jax.numpy as jnp

import pytreeclass as pytc
import pytreeclass._src.dataclass_util as dcu

# from pytreeclass._src.dispatch import dispatch
from pytreeclass._src.dataclass_util import _dataclass_structure
from pytreeclass.tree_viz.box_drawing import _table
from pytreeclass.tree_viz.node_pprint import _format_node_repr
from pytreeclass.tree_viz.utils import (
    _reduce_count_and_size,
    _sequential_tree_shape_eval,
)

PyTree = Any


def _format_size(node_size, newline=False):
    """return formatted size from inexact(exact) complex number"""
    mark = "\n" if newline else ""
    order_kw = ["B", "KB", "MB", "GB"]

    if isinstance(node_size, complex):
        # define order of magnitude
        real_size_order = int(math.log(max(node_size.real, 1), 1024))
        imag_size_order = int(math.log(max(node_size.imag, 1), 1024))
        fmt = f"{(node_size.real)/(1024**real_size_order):.2f}{order_kw[real_size_order]}{mark}"
        fmt += f"({(node_size.imag)/(1024**imag_size_order):.2f}{order_kw[imag_size_order]})"
        return fmt

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
        tree: dataclass decorated class.
        array: Input jax.numpy used to call the class. Defaults to None.
        show_type: Whether to print the type column. Defaults to True.
        show_param: Whether to print the parameter column. Defaults to True.
        show_size: Whether to print the size column. Defaults to True.
        show_config: Whether to print the config column. Defaults to True.

    Example:
        @pytc.treeclass
        class Test:
            a: int = 0
            b : jnp.ndarray = jnp.array([1,2,3])
            c : float = 1.0

        >>> print(tree_summary(Test(config=True)))
        ┌────┬───────────┬───────┬─────────────┬─────────────────────────────┐
        │Name│Type       │Param #│Size         │Config                       │
        ├────┼───────────┼───────┼─────────────┼─────────────────────────────┤
        │a   │int        │0(1)   │0.00B(24.00B)│a=0                          │
        ├────┼───────────┼───────┼─────────────┼─────────────────────────────┤
        │b   │DeviceArray│0(3)   │0.00B(12.00B)│b=i32[3]∈[1,3]<μ=2.00,σ=0.82>│
        ├────┼───────────┼───────┼─────────────┼─────────────────────────────┤
        │c   │float      │1(0)   │24.00B(0.00B)│c=1.0                        │
        └────┴───────────┴───────┴─────────────┴─────────────────────────────┘
        Total count:	1(4)
        Trainable count:1(4)
        Frozen count:	0(0)
        ----------------------------------------------------------------------
        Total size:	24.00B(36.00B)
        Trainable size:	24.00B(36.00B)
        Frozen size:	0.00B(0.00B)
        ======================================================================

    Note:
        values inside () defines the info about the non-inexact (i.e.) non-differentiable parameters.
        this distinction is important for the jax.grad function.
        to see which values types needs to be handled for training

    Returns:
        str: Summary of the tree structure.
    """
    _format_node = lambda node: _format_node_repr(node).expandtabs(1)

    if array is not None:
        # run through the tree to get the shape of the tree
        shape = _sequential_tree_shape_eval(tree, array)
        indim_shape, outdim_shape = shape[:-1], shape[1:]

        in_shape_str = ["Input"] + list(map(_format_node, indim_shape))
        out_shape_str = ["Output"] + list(map(_format_node, outdim_shape))

    def recurse_field(field_item, node_item, name_path, type_path):

        nonlocal ROWS, COUNT, SIZE

        if not field_item.repr:
            # skip the current field if field(repr=False)
            return

        if isinstance(node_item, (list, tuple)) and any(
            dc.is_dataclass(leaf) for leaf in node_item
        ):
            # case of a leaf container
            # expand container if any item is a `dataclass`
            for i, layer in enumerate(node_item):

                if dcu.is_field_frozen(field_item):
                    # all the items in the container are frozen if the container is frozen
                    new_field = dc.field(metadata={"static": "frozen"})
                else:
                    # create a new field for each item in the container
                    new_field = dc.field()

                object.__setattr__(new_field, "name", f"{field_item.name}[{i}]")
                object.__setattr__(new_field, "type", type(layer))
                object.__setattr__(new_field, "_field_type", dc._FIELD)

                recurse_field(
                    field_item=new_field,
                    node_item=layer,
                    name_path=name_path + (f"{field_item.name}[{i}]",),
                    type_path=type_path + (layer.__class__.__name__,),
                )

        elif dc.is_dataclass(node_item):
            # check if the node is frozen or it all the fields are frozen
            is_frozen = dcu.is_field_frozen(field_item)
            is_frozen = is_frozen or dcu.is_dataclass_fields_frozen(node_item)
            dynamic, _ = _dataclass_structure(pytc.tree_unfilter(node_item))
            count, size = _reduce_count_and_size(pytc.tree_unfilter(node_item))

            if is_frozen:
                row = ["/".join(name_path) + "\n(frozen)"]
            else:
                row = ["/".join(name_path)]

            row += ["/".join(type_path)] if show_type else []
            row += [_format_count(count)] if show_param else []
            row += [_format_size(size)] if show_size else []

            if show_config:
                row += [
                    "\n".join([f"{k}={_format_node(v)}" for k, v in dynamic.items()])
                ]
            else:
                row += []

            ROWS.append(row)

            for field_item in dc.fields(node_item):
                is_frozen = dcu.is_field_frozen(field_item)
                COUNT[1 if is_frozen else 0] += count
                SIZE[1 if is_frozen else 0] += size

        else:
            # case of a leaf parameter
            if dcu.is_field_nondiff(field_item):
                return

            is_nondiff = dcu.is_field_frozen(field_item)
            count, size = _reduce_count_and_size(node_item)

            if is_nondiff:
                row = ["/".join(name_path) + "(frozen)"]
            else:
                row = ["/".join(name_path)]

            row += ["/".join(type_path)] if show_type else []
            row += [_format_count(count)] if show_param else []
            row += [_format_size(size)] if show_size else []
            row += [f"{field_item.name}={_format_node(node_item)}"] if show_config else []  # fmt: skip

            ROWS.append(row)

            # non-treeclass leaf inherit frozen state
            COUNT[1 if is_nondiff else 0] += count
            SIZE[1 if is_nondiff else 0] += size

    def recurse(tree, name_path, type_path):

        nonlocal ROWS, COUNT, SIZE

        for field_item in dc.fields(tree):
            node_item = getattr(tree, field_item.name)

            # non-leaf dataclass
            if dcu.is_dataclass_non_leaf(node_item):
                recurse(
                    tree=node_item,
                    name_path=name_path + (field_item.name,),
                    type_path=type_path + (node_item.__class__.__name__,),
                )

            else:
                # skip nondiff fields
                if not dcu.is_field_nondiff(field_item):
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
        f"Total count:\t{_format_count(sum(COUNT))}\n"
        f"Trainable count:{_format_count(COUNT[0])}\n"
        f"Frozen count:\t{_format_count(COUNT[1])}\n"
        f"{'-'*max([table_width,40])}\n"
        f"Total size:\t{_format_size(sum(SIZE))}\n"
        f"Trainable size:\t{_format_size(SIZE[0])}\n"
        f"Frozen size:\t{_format_size(SIZE[1])}\n"
        f"{'='*max([table_width,40])}"
    )

    return layer_table + "\n" + (param_summary)
