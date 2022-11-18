from __future__ import annotations

import copy
import dataclasses
from collections.abc import Iterable
from types import FunctionType
from typing import Any, Callable

import jax.numpy as jnp
import jax.tree_util as jtu

import pytreeclass._src.dataclass_util as dcu

PyTree = Any


def _tree_callable_filter(tree, where: Callable[[Any], bool], unfilter: bool = False):
    """Filter a tree based on a callable function.

    Args:
        tree: The tree to filter.
        where: A callable function that takes a node and returns a boolean.
        unfilter: If True, unfilter the tree. Removes the static metadata from the tree.
    """
    _dataclass_fields = dict(tree.__dataclass_fields__)
    for name in _dataclass_fields:
        field_item = _dataclass_fields[name]
        node_item = getattr(tree, name)

        if dataclasses.is_dataclass(node_item):
            _tree_callable_filter(node_item, where, unfilter)

        elif where(node_item):
            field_item = dcu.field_copy(field_item)
            _meta = dict(field_item.metadata)

            if unfilter and _meta.get("static", False) == "frozen":
                # remove frozen static metadata
                del _meta["static"]

            elif not unfilter and not _meta.get("static", False):
                # add static metadata if not already present
                _meta["static"] = "frozen"

            field_item.metadata = _meta
            _dataclass_fields[name] = field_item

    object.__setattr__(tree, "__dataclass_fields__", _dataclass_fields)
    return tree


def _tree_pytree_filter(tree: PyTree, where: PyTree, unfilter: bool = False):
    """Filter a tree based on a pytree of booleans.

    Args:
        tree: The tree to filter.
        where: A pytree of booleans.
        unfilter: If True, unfilter the tree. Removes the static metadata from the tree.
    """
    _dataclass_fields = dict(tree.__dataclass_fields__)
    for name in _dataclass_fields:
        field_item = _dataclass_fields[name]
        node_item = getattr(tree, name)
        node_filter = getattr(where, name)

        if dataclasses.is_dataclass(node_item):
            _tree_pytree_filter(node_item, node_filter, unfilter)

        elif dcu.is_dataclass_leaf_bool(node_filter) and jnp.all(node_filter):
            field_item = dcu.field_copy(field_item)
            _meta = dict(field_item.metadata)

            if unfilter and _meta.get("static", False) == "frozen":
                # remove frozen static metadata
                del _meta["static"]

            elif not unfilter and not _meta.get("static", False):
                # add static metadata if not already present
                _meta["static"] = "frozen"

            field_item.metadata = _meta
            _dataclass_fields[name] = field_item

    object.__setattr__(tree, "__dataclass_fields__", _dataclass_fields)
    return tree


def _tree_filter(
    tree: PyTree, where: Callable[[Any], bool] | PyTree, unfilter: bool = False
):
    """Filter a tree based on a callable function or a pytree of booleans.
    The function modies the tree metadata static flag where is True.

    Args:
        tree: The tree to filter.
        where: A callable function or a pytree of booleans.
        unfilter: If True, unfilter the tree. Removes the static metadata from the tree.
    """
    if isinstance(where, FunctionType):
        return _tree_callable_filter(copy.copy(tree), where, unfilter=unfilter)
    elif isinstance(where, type(tree)):
        return _tree_pytree_filter(copy.copy(tree), where, unfilter=unfilter)
    msg = f"filter must be a function or a tree of the same type as tree, got {type(filter)}"
    raise TypeError(msg)


def is_nondiff(item: Any) -> bool:
    """Check if a node is non-differentiable."""

    def _is_nondiff_item(node: Any):
        if isinstance(node, (float, complex)) or dataclasses.is_dataclass(node):
            return False
        elif isinstance(node, jnp.ndarray) and jnp.issubdtype(node.dtype, jnp.inexact):
            return False

        return True

    if isinstance(item, Iterable):
        # if an iterable has at least one non-differentiable item
        # then the whole iterable is non-differentiable
        return any([_is_nondiff_item(item) for item in jtu.tree_leaves(item)])
    return _is_nondiff_item(item)


def tree_filter(tree: PyTree, *, where: Callable[[Any], bool] | PyTree = None):
    """Filter a tree based on a callable function or a pytree of booleans.

    Args:
        tree: The tree to filter.
        where: A callable function or a pytree of booleans.
    """
    return _tree_filter(tree=tree, where=(where or is_nondiff))


def tree_unfilter(tree: PyTree, *, where: Callable[[Any], bool] | PyTree = None):
    """Unfilter a tree based on a callable function or a pytree of booleans.

    Args:
        tree: The tree to unfilter.
        where: A callable function or a pytree of booleans.
    """
    return _tree_filter(tree=tree, where=(where or (lambda _: True)), unfilter=True)
