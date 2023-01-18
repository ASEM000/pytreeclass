from __future__ import annotations

import copy
import dataclasses as dc
from types import FunctionType
from typing import Any, Callable

import numpy as np

import pytreeclass as pytc
from pytreeclass._src.utils import _mutable, is_nondiff

PyTree = Any


@_mutable
def _tree_filter_unfilter(
    tree: PyTree, where: PyTree | FunctionType, filter: bool
) -> PyTree:
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
    # this function change a Field to FilteredField (i.e. filtered Field) if where condition is met and filter is True
    # if filter is false then it changes FilteredField to Field (i.e. unfiltered Field) if where condition is met
    def _filter_unfilter(tree: PyTree, where: PyTree | Callable):
        _dataclass_fields = dict(tree.__dataclass_fields__)

        for name in _dataclass_fields:
            node_item = getattr(tree, name)
            field_item = _dataclass_fields[name]

            if dc.is_dataclass(node_item):
                # in case of non-leaf recurse deeper
                where = getattr(where, name) if isinstance(where, type(tree)) else where
                _filter_unfilter(tree=node_item, where=where)
                continue

            # leaf case where can be either a bool tree leaf of a function
            node_where = (
                getattr(where, name)
                if isinstance(where, type(tree))
                else where(node_item)
            )

            # check if the where leaf is a bool
            if not isinstance(node_where, bool) and not (
                hasattr(node_where, "dtype") and node_where.dtype == "bool"
            ):
                if isinstance(field_item, pytc.NonDiffField):
                    # skip if the field is already a NonDiffField
                    continue

                # raise error if the where leaf is not a bool, and the field is not a NonDiffField
                msg = f"Non bool leaf found in where leaf. Found {(node_where)}"
                raise TypeError(msg)

            # continue if the where leaf is False
            if not np.all(node_where):
                continue

            # if the where condition is True, then we need to filter/undo the filtering

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
                if not isinstance(field_item, pytc.NonDiffField):
                    # if we are filtering and the field is not already a NonDiffField
                    field_item = pytc.FilteredField(**field_params)
            else:
                if isinstance(field_item, pytc.FilteredField):
                    # if we are unfiltering and the field is a FilteredField
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
        return _filter_unfilter(copy.copy(tree), where)
    raise TypeError("Where must be of same type as `tree`  or a `Callable`")


def tree_filter(tree: PyTree, *, where: Callable[[Any], bool] | PyTree = None):
    """Filter a tree based on a callable function or a pytree of booleans.

    Args:
        tree: The tree to filter.
        where: A callable function or a pytree of booleans. Defaults to filtering non-differentiable nodes.
    """
    if not dc.is_dataclass(tree):
        raise TypeError("Tree must be a dataclass")
    return _tree_filter_unfilter(tree, where=(where or is_nondiff), filter=True)


def tree_unfilter(tree: PyTree, *, where: Callable[[Any], bool] | PyTree = None):
    """Unfilter a tree based on a callable function or a pytree of booleans.

    Args:
        tree: The tree to unfilter.
        where: A callable function or a pytree of booleans. Defaults to unfiltering all nodes.
    """
    if not dc.is_dataclass(tree):
        raise TypeError("Tree must be a dataclass")
    return _tree_filter_unfilter(tree, where=(where or (lambda _: True)), filter=False)
