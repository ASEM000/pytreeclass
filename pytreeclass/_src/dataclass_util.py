from __future__ import annotations

import copy
import dataclasses as dc
import functools as ft
from types import FunctionType
from typing import Any, Callable, Iterable

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

import pytreeclass as pytc

PyTree = Any


def _set_dataclass_frozen(tree: PyTree, frozen: bool) -> PyTree:
    """Set frozen to for a dataclass instance"""
    if dc.is_dataclass(tree):
        setattr(tree.__dataclass_params__, "frozen", frozen)
        for field_item in dc.fields(tree):
            if hasattr(tree, field_item.name):
                _set_dataclass_frozen(getattr(tree, field_item.name), frozen)
    return tree


def _mutable(func):
    """decorator that allow mutable behvior
    for class methods/ function with treeclass as first argument

    Example:
        class ... :

        >>> @_mutable
        ... def mutable_method(self):
        ...    return self.value + 1
    """
    msg = f"`mutable` can only be applied to methods. Found{type(func)}"
    assert isinstance(func, FunctionType), msg

    @ft.wraps(func)
    def mutable_method(self, *a, **k):
        self = _set_dataclass_frozen(tree=self, frozen=False)
        output = func(self, *a, **k)
        self = _set_dataclass_frozen(tree=self, frozen=True)
        return output

    return mutable_method


@_mutable
def _tree_filter(tree: PyTree, where: PyTree | FunctionType, filter: bool) -> PyTree:
    """Filter a tree based on a where condition.

    Args:
        tree: The tree to filter.
        where: The where condition.
        filter: If True filter the tree, else undo the filtering

    Returns:
        The filtered tree.
    """
    # this function mutate the dataclass fields that is stored in metadata of the tree
    # the flatten function ignores values with NonDiffField type,
    # this function change a Field to FrozenField (i.e. filtered Field) if where condition is met and filter is True
    # if filter is false then it changes FrozenField to Field (i.e. unfiltered Field) if where condition is met

    def _filter(tree: PyTree, where: PyTree):
        _dataclass_fields = dict(tree.__dataclass_fields__)

        for name in _dataclass_fields:
            node_item = getattr(tree, name)

            if dc.is_dataclass(node_item):
                # in case of non-leaf recurse deeper
                where = getattr(where, name) if isinstance(where, type(tree)) else where
                _filter(tree=node_item, where=where)

            else:
                # leaf case
                # where can be either a bool tree leaf of a function
                if isinstance(where, type(tree)):
                    node_where = getattr(where, name)
                else:
                    # where is a function
                    node_where = where(node_item)

                # check if the where condition is a bool tree leaf
                # or a bool array and if all elements are True
                if (
                    isinstance(node_where, bool)
                    or (hasattr(node_where, "dtype") and node_where.dtype == "bool")
                ) and np.all(node_where):

                    # if the where condition is True, then we need to filter/undo the filtering
                    field_item = _dataclass_fields[name]
                    field_name, field_type = field_item.name, field_item.type

                    # extract the field parameters to create a new field
                    # either frozen-> non-frozen or non-frozen -> frozen
                    field_params = dict(
                        default=field_item.default,
                        default_factory=field_item.default_factory,
                        init=field_item.init,
                        repr=field_item.repr,
                        hash=field_item.hash,
                        compare=field_item.compare,
                        metadata=field_item.metadata,
                    )

                    # change this once py requirement is 3.10+
                    if "kw_only" in dir(field_item):
                        field_params.update(kw_only=field_item.kw_only)

                    if filter:
                        # transform the field to a frozen field iff it is not a NonDiffField
                        if not isinstance(field_item, pytc.NonDiffField):
                            # convert to a frozen field
                            field_item = pytc.FrozenField(**field_params)
                            object.__setattr__(field_item, "name", field_name)
                            object.__setattr__(field_item, "type", field_type)
                            object.__setattr__(field_item, "_field_type", dc._FIELD)

                    else:
                        # transform the field to a default field iff it is a FrozenField
                        # in essence we want to undo the filtering, so we need to convert the filtered fields
                        # (i.e. FrozenField) to default fields (i.e. dc.Field)
                        if isinstance(field_item, pytc.FrozenField):
                            # convert to a default field
                            field_item = dc.Field(**field_params)
                            object.__setattr__(field_item, "name", field_name)
                            object.__setattr__(field_item, "type", field_type)
                            object.__setattr__(field_item, "_field_type", dc._FIELD)

                    # update the fields dict
                    _dataclass_fields[name] = field_item

        # update the tree fields of the tree at the current level
        setattr(tree, "__dataclass_fields__", _dataclass_fields)
        return tree

    # check if where is a callable or a tree of the same type as tree
    # inside the _filter function we will check if where is a tree leafs are bools
    if isinstance(where, FunctionType) or isinstance(where, type(tree)):
        # we got to copy the tree to avoid mutating the original tree
        return _filter(copy.copy(tree), where)
    raise TypeError("Where must be of same type as `tree`  or a `Callable`")


def is_nondiff(item: Any) -> bool:
    """Check if a node is non-differentiable."""

    def _is_nondiff_item(node: Any):
        if (
            (hasattr(node, "dtype") and jnp.issubdtype(node.dtype, jnp.inexact))
            or isinstance(node, (float, complex))
            or dc.is_dataclass(node)
        ):
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
        where: A callable function or a pytree of booleans. Defaults to filtering non-differentiable nodes.
    """
    if not dc.is_dataclass(tree):
        raise TypeError("Tree must be a dataclass")
    return _tree_filter(tree, where=(where or is_nondiff), filter=True)


def tree_unfilter(tree: PyTree, *, where: Callable[[Any], bool] | PyTree = None):
    """Unfilter a tree based on a callable function or a pytree of booleans.

    Args:
        tree: The tree to unfilter.
        where: A callable function or a pytree of booleans. Defaults to unfiltering all nodes.
    """
    if not dc.is_dataclass(tree):
        raise TypeError("Tree must be a dataclass")
    return _tree_filter(tree, where=(where or (lambda _: True)), filter=False)
