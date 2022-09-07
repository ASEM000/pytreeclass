from __future__ import annotations

import math
import os
from dataclasses import field

import jax.numpy as jnp

import pytreeclass._src as src
from pytreeclass._src.dispatch import dispatch
from pytreeclass._src.tree_util import (
    _tree_fields,
    _tree_structure,
    is_treeclass_frozen,
    is_treeclass_non_leaf,
    tree_unfreeze,
)
from pytreeclass.tree_viz.box_drawing import _table
from pytreeclass.tree_viz.node_pprint import _format_node_repr
from pytreeclass.tree_viz.utils import (
    _reduce_count_and_size,
    _sequential_tree_shape_eval,
)


def _format_size(node_size, newline=False):
    """return formatted size from inexact(exact) complex number"""
    mark = "\n" if newline else ""
    order_kw = ["B", "KB", "MB", "GB"]

    # define order of magnitude
    real_size_order = int(math.log(node_size.real, 1024)) if node_size.real > 0 else 0
    imag_size_order = int(math.log(node_size.imag, 1024)) if node_size.imag > 0 else 0
    return (
        f"{(node_size.real)/(1024**real_size_order):.2f}{order_kw[real_size_order]}{mark}"
        f"({(node_size.imag)/(1024**imag_size_order):.2f}{order_kw[imag_size_order]})"
    )


def _format_count(node_count, newline=False):
    mark = "\n" if newline else ""
    return f"{int(node_count.real):,}{mark}({int(node_count.imag):,})"


def tree_summary(tree, array: jnp.ndarray = None) -> str:
    """Prints a summary of the tree structure.

    Args:
        tree (PyTree): @treeclass decorated class.
        array (jnp.ndarray, optional): Input jax.numpy used to call the class. Defaults to None.

    Example:
        @pytc.treeclass
        class Test:
            a: int = 0
            b : jnp.ndarray = jnp.array([1,2,3])
            c : float = 1.0

        >>> print(tree_summary(Test()))
        ┌────┬───────────┬───────┬────────┬────────┐
        │Name│Type       │Param #│Size    │Config  │
        ├────┼───────────┼───────┼────────┼────────┤
        │a   │int        │0(1)   │0.00B   │a=0     │
        │    │           │       │(24.00B)│        │
        ├────┼───────────┼───────┼────────┼────────┤
        │b   │DeviceArray│0(3)   │0.00B   │b=i32[3]│
        │    │           │       │(12.00B)│        │
        ├────┼───────────┼───────┼────────┼────────┤
        │c   │float      │1(0)   │24.00B  │c=1.0   │
        │    │           │       │(0.00B) │        │
        └────┴───────────┴───────┴────────┴────────┘
        Total count :	1(4)
        Dynamic count :	1(4)
        Frozen count :	0(0)
        --------------------------------------------
        Total size :	24.00B(36.00B)
        Dynamic size :	24.00B(36.00B)
        Frozen size :	0.00B(0.00B)
        ============================================

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

        shape_str = ["Input/Output"] + [
            f"{_format_node(indim_shape[i])}\n{_format_node(outdim_shape[i])}"
            for i in range(len(indim_shape))
        ]

    @dispatch(argnum="node_item")
    def recurse_field(field_item, node_item, is_frozen, name_path, type_path):
        ...

    @recurse_field.register(int)
    @recurse_field.register(float)
    @recurse_field.register(complex)
    @recurse_field.register(str)
    @recurse_field.register(bool)
    @recurse_field.register(jnp.ndarray)
    def _(field_item, node_item, is_frozen, name_path, type_path):

        nonlocal ROWS, COUNT, SIZE

        if field_item.repr:
            count, size = _reduce_count_and_size(node_item)
            ROWS.append(
                [
                    "/".join(name_path) + f"{('(frozen)' if is_frozen else '')}",
                    "/".join(type_path),
                    _format_count(count),
                    _format_size(size, True),
                    f"{field_item.name}={_format_node(node_item)}",
                ]
            )

            # non-treeclass leaf inherit frozen state
            COUNT[1 if is_frozen else 0] += count
            SIZE[1 if is_frozen else 0] += size

    @recurse_field.register(list)
    @recurse_field.register(tuple)
    def _(field_item, node_item, is_frozen, name_path, type_path):
        # handles containers
        # here what we do is we just add the name/type of the container to the path by passing
        # a created field_item with the name/type for each item in the container
        if field_item.repr:

            for i, layer in enumerate(node_item):
                new_field = field()
                object.__setattr__(new_field, "name", f"{field_item.name}_{i}")
                object.__setattr__(new_field, "type", type(layer))

                recurse_field(
                    field_item=new_field,
                    node_item=layer,
                    is_frozen=is_frozen,
                    name_path=name_path + (f"{field_item.name}_{i}",),
                    type_path=type_path + (layer.__class__.__name__,),
                )

    @recurse_field.register(src.tree_base._treeBase)
    def _(field_item, node_item, is_frozen, name_path, type_path):
        # handles treeclass
        nonlocal ROWS, COUNT, SIZE

        if field_item.repr:
            is_frozen = is_treeclass_frozen(node_item)
            count, size = _reduce_count_and_size(tree_unfreeze(node_item))
            dynamic, _ = _tree_structure(node_item)
            ROWS.append(
                [
                    "/".join(name_path)
                    + f"{(os.linesep + '(frozen)' if is_frozen else '')}",
                    "/".join(type_path),
                    _format_count(count),
                    _format_size(size, True),
                    "\n".join([f"{k}={_format_node(v)}" for k, v in dynamic.items()]),
                ]
            )

            COUNT[1 if is_frozen else 0] += count
            SIZE[1 if is_frozen else 0] += size

    def recurse(tree, is_frozen, name_path, type_path):

        nonlocal ROWS, COUNT, SIZE

        for field_item in _tree_fields(tree).values():

            node_item = tree.__dict__[field_item.name]

            if is_treeclass_non_leaf(node_item):
                # recurse if the field is a treeclass
                # the recursion passes the frozen state of the current node
                # name_path,type_path (i.e. location of the ndoe in the tree)
                # for instance a path "L1/L0" defines a class L0 with L1 parent
                recurse(
                    tree=node_item,
                    is_frozen=is_treeclass_frozen(node_item),
                    name_path=name_path + (field_item.name,),
                    type_path=type_path + (node_item.__class__.__name__,),
                )

            else:

                is_static = field_item.metadata.get("static", False)
                # skip if the field is static

                if not (is_static):
                    recurse_field(
                        field_item=field_item,
                        node_item=node_item,
                        is_frozen=is_frozen,
                        name_path=name_path + (field_item.name,),
                        type_path=type_path + (node_item.__class__.__name__,),
                    )

    ROWS = [["Name", "Type ", "Param #", "Size ", "Config"]]
    COUNT = [0, 0]
    SIZE = [0, 0]

    recurse(tree, is_frozen=is_treeclass_frozen(tree), name_path=(), type_path=())

    # we need to transform rows to cols
    # as `_table` concatenates columns together
    COLS = [list(c) for c in zip(*ROWS)]

    if array is not None:
        COLS += [shape_str]

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

    return layer_table + "\n" + param_summary
