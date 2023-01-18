from __future__ import annotations

import dataclasses as dc
import math
import sys
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

import pytreeclass as pytc
import pytreeclass._src.utils as dcu


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

    elif isinstance(node_size, (float, int)):
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
    elif isinstance(node_count, (float, int)):
        return f"{int(node_count):,}"

    raise TypeError(f"node_count must be int or float, got {type(node_count)}")


def _is_dataclass_fields_nondiff(tree):
    """assert if a dataclass is static"""
    if dc.is_dataclass(tree):
        field_items = dc.fields(tree)
        if len(field_items) > 0:
            return all(isinstance(f, pytc.NonDiffField) for f in field_items)
    return False


def _is_dataclass_fields_frozen(tree):
    """assert if a dataclass is static"""
    if dc.is_dataclass(tree):
        field_items = dc.fields(tree)
        if len(field_items) > 0:
            return all(isinstance(f, pytc.FilteredField) for f in field_items)
    return False


def _is_dataclass_leaf(tree):
    """assert if a node is dataclass leaf"""
    if dc.is_dataclass(tree):

        return dc.is_dataclass(tree) and not any(
            [dc.is_dataclass(getattr(tree, fi.name)) for fi in dc.fields(tree)]
        )
    return False


def _is_dataclass_non_leaf(tree):
    return dc.is_dataclass(tree) and not _is_dataclass_leaf(tree)


def _mermaid_marker(field_item: dc.Field, node_item: Any, default: str = "--") -> str:
    """return the suitable marker given the field and node item

    Args:
        field_item (Field): field item of the pytree node
        node_item (Any): node item
        default (str, optional): default marker. Defaults to "".

    Returns:
        str: marker character.
    """
    # for now, we only have two markers '*' for non-diff and '#' for frozen
    if isinstance(field_item, pytc.FilteredField) or _is_dataclass_fields_frozen(
        node_item
    ):
        return "-..-"

    if isinstance(field_item, pytc.NonDiffField) or _is_dataclass_fields_nondiff(
        node_item
    ):
        return "--x"

    return default


def _marker(field_item: dc.Field, node_item: Any, default: str = "") -> str:
    """return the suitable marker given the field and node item"""
    # '*' for non-diff

    if isinstance(field_item, pytc.FilteredField) or _is_dataclass_fields_frozen(
        node_item
    ):
        return "#"

    if isinstance(field_item, pytc.NonDiffField) or _is_dataclass_fields_nondiff(
        node_item
    ):
        return "*"

    return default


def _sequential_tree_shape_eval(tree, array):
    """Evaluate shape propagation of assumed sequential modules"""
    dyanmic, static = dcu._dataclass_structure(tree)

    # all dynamic/static leaves
    all_leaves = (*dyanmic.values(), *static.values())
    leaves = [leaf for leaf in all_leaves if dc.is_dataclass(leaf)]

    shape = [jax.eval_shape(lambda x: x, array)]
    for leave in leaves:
        shape += [jax.eval_shape(leave, shape[-1])]
    return shape


def _reduce_count_and_size(leaf):
    """reduce params count and params size of a tree of leaves to be used in `tree_summary`"""

    def reduce_func(acc, node):
        lhs_count, lhs_size = acc
        rhs_count, rhs_size = _node_count_and_size(node)
        return (lhs_count + rhs_count, lhs_size + rhs_size)

    return jtu.tree_reduce(reduce_func, leaf, (complex(0, 0), complex(0, 0)))


def _node_count_and_size(node: Any) -> tuple[complex, complex]:
    """Calculate number and size of `trainable` and `non-trainable` parameters

    Returns:
        complex: Complex number of (inexact, exact) parameters for count/size
    """

    if isinstance(node, (jnp.ndarray, np.ndarray)):

        if jnp.issubdtype(node.dtype, jnp.inexact):
            # inexact(trainable) array
            count = complex(int(jnp.array(node.shape).prod()), 0)
            size = complex(int(node.nbytes), 0)
        else:
            # non in-exact paramter
            count = complex(0, int(jnp.array(node.shape).prod()))
            size = complex(0, int(node.nbytes))

    elif isinstance(node, (float, complex)):
        # inexact non-array (array_like)
        count = complex(1, 0)
        size = complex(sys.getsizeof(node), 0)

    elif isinstance(node, int):
        # exact non-array
        count = complex(0, 1)
        size = complex(0, sys.getsizeof(node))
        return count, size

    else:
        count = complex(0, 1)
        size = complex(0, sys.getsizeof(node))

    return count, size
