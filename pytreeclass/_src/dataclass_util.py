from __future__ import annotations

import copy
import dataclasses as dc
import functools as ft
from types import FunctionType
from typing import Any, Callable

import numpy as np

PyTree = Any


class _fieldDict(dict):
    # using a regular dict will cause the following error:
    # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def _dataclass_structure(tree) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return dynamic and static fields of a dataclass instance based on metadata static flag"""
    static, dynamic = _fieldDict(tree.__dict__), _fieldDict()

    for field_item in dc.fields(tree):
        if field_item.metadata.get("static", False) is False:
            dynamic[field_item.name] = static.pop(field_item.name)

    return (dynamic, static)


def _dataclass_unfreeze(tree):
    """Set frozen to False for a dataclass instance"""
    if dc.is_dataclass(tree):
        object.__setattr__(tree.__dataclass_params__, "frozen", False)
        for field_item in dc.fields(tree):
            if hasattr(tree, field_item.name):
                _dataclass_unfreeze(getattr(tree, field_item.name))
    return tree


def _dataclass_freeze(tree):
    """Set frozen to True for a dataclass instance"""
    if dc.is_dataclass(tree):
        object.__setattr__(tree.__dataclass_params__, "frozen", True)
        for field_item in dc.fields(tree):
            if hasattr(tree, field_item.name):
                _dataclass_freeze(getattr(tree, field_item.name))
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
        self = _dataclass_unfreeze(tree=self)
        output = func(self, *a, **k)
        self = _dataclass_freeze(tree=self)
        return output

    return mutable_method


def field(*, nondiff: bool = False, **k) -> dc.Field:
    """Similar to dc.field but with additional arguments
    Args:
        nondiff: if True, the field will not be differentiated or modified by any filtering operations
        name: name of the field. Will be inferred from the variable name if its assigned to a class attribute.
        type: type of the field. Will be inferred from the variable type if its assigned to a class attribute.
        **k: additional arguments to pass to dc.field
    """
    metadata = k.pop("metadata", {})
    if nondiff is True:
        metadata["static"] = "nondiff"

    return dc.field(metadata=metadata, **k)


def field_copy(field_item):
    """Copy a dataclass field item."""
    new_field = dc.field(
        default=field_item.default,
        default_factory=field_item.default_factory,
        init=field_item.init,
        repr=field_item.repr,
        hash=field_item.hash,
        compare=field_item.compare,
        metadata=dict(field_item.metadata),
    )

    object.__setattr__(new_field, "name", field_item.name)
    object.__setattr__(new_field, "type", field_item.type)
    object.__setattr__(new_field, "_field_type", field_item._field_type)

    return new_field


def is_field_nondiff(field_item: dc.Field) -> bool:
    """check if field is strictly static"""
    return (
        isinstance(field_item, dc.Field)
        and field_item.metadata.get("static", False) == "nondiff"
    )


def is_field_frozen(field_item: dc.Field) -> bool:
    """check if field is strictly static"""
    return (
        isinstance(field_item, dc.Field)
        and field_item.metadata.get("static", False) == "frozen"
    )


def is_dataclass_fields_nondiff(tree):
    """assert if a dataclass is static"""
    if dc.is_dataclass(tree):
        field_items = dc.fields(tree)
        if len(field_items) > 0:
            return all(is_field_nondiff(f) for f in field_items)
    return False


def is_dataclass_fields_frozen(tree):
    """assert if a dataclass is static"""
    if dc.is_dataclass(tree):
        field_items = dc.fields(tree)
        if len(field_items) > 0:
            return all(is_field_frozen(f) for f in field_items)
    return False


def is_leaf_bool(node):
    """assert if a dataclass leaf is boolean (for boolen indexing)"""
    if hasattr(node, "dtype"):
        return node.dtype == "bool"
    return isinstance(node, bool)


def is_dataclass_leaf(tree):
    """assert if a node is dataclass leaf"""
    if dc.is_dataclass(tree):

        return dc.is_dataclass(tree) and not any(
            [dc.is_dataclass(getattr(tree, fi.name)) for fi in dc.fields(tree)]
        )
    return False


def is_dataclass_non_leaf(tree):
    return dc.is_dataclass(tree) and not is_dataclass_leaf(tree)


def dataclass_leaves(tree):
    """return all leaves of a dataclass"""

    def _recurse(tree):
        for node_item in (
            [f, getattr(tree, f.name)]
            for f in dc.fields(tree)
            if not f.metadata.get("static", False)
        ):
            if dc.is_dataclass(node_item):
                yield from _recurse(node_item)
            else:
                yield node_item

    if dc.is_dataclass(tree):
        return list(_recurse(tree))
    raise TypeError("tree must be a dataclass")


def dataclass_reduce(function: Callable, tree: Any, initializer: Any = None):
    """reduce a dataclass tree. Similar to jtu.tree_reduce but for dataclasses"""
    if initializer is None:
        return ft.reduce(function, dataclass_leaves(tree))
    return ft.reduce(function, dataclass_leaves(tree), initializer)


def dataclass_filter(tree: PyTree, where: PyTree | FunctionType):
    """Filter a dataclass based on a boolean leaf dataclass.
    This function works by adding a static metadata to the dataclass fields that are filtered out.

    Args:
        tree: The tree to filter.
        where: A dataclass of boolean leaves or a function that takes a leaf and returns a boolean.
    """

    def _dataclass_filter(tree: PyTree, where: PyTree):
        _dataclass_fields = dict(tree.__dataclass_fields__)

        for name in _dataclass_fields:
            node_item = getattr(tree, name)

            if dc.is_dataclass(node_item):
                where = getattr(where, name) if isinstance(where, type(tree)) else where
                _dataclass_filter(tree=node_item, where=where)

            else:
                node_where = (
                    getattr(where, name)
                    if isinstance(where, type(tree))
                    else where(node_item)
                )

                if (
                    is_leaf_bool(node_where)
                    and np.all(node_where)
                    and not _dataclass_fields[name].metadata.get("static", False)
                ):
                    field_item = field_copy(_dataclass_fields[name])
                    _meta = dict(field_item.metadata)
                    _meta["static"] = "frozen"
                    field_item.metadata = _meta
                    _dataclass_fields[name] = field_item

        object.__setattr__(tree, "__dataclass_fields__", _dataclass_fields)
        return tree

    if isinstance(where, FunctionType) or isinstance(where, type(tree)):
        return _dataclass_filter(copy.copy(tree), where)
    raise TypeError("Where must be of same type as `tree`  or a `Callable`")


def dataclass_unfilter(tree: PyTree, where: PyTree | FunctionType):
    """Filter a dataclass based on a boolean leaf dataclass.
    This function works by adding a static metadata to the dataclass fields that are filtered out.

    Args:
        tree: The tree to filter.
        where: A dataclass of boolean leaves or a function that takes a leaf and returns a boolean.
    """

    def _dataclass_unfilter(tree: PyTree, where: PyTree):
        _dataclass_fields = dict(tree.__dataclass_fields__)

        for name in _dataclass_fields:
            node_item = getattr(tree, name)

            if dc.is_dataclass(node_item):
                where = getattr(where, name) if isinstance(where, type(tree)) else where
                _dataclass_unfilter(tree=node_item, where=where)

            else:
                node_where = (
                    where(node_item)
                    if isinstance(where, FunctionType)
                    else getattr(where, name)
                )

                if (
                    is_leaf_bool(node_where)
                    and np.all(node_where)
                    and _dataclass_fields[name].metadata.get("static", False)
                    == "frozen"
                ):
                    field_item = field_copy(_dataclass_fields[name])
                    _meta = dict(field_item.metadata)
                    del _meta["static"]
                    field_item.metadata = _meta
                    _dataclass_fields[name] = field_item

        object.__setattr__(tree, "__dataclass_fields__", _dataclass_fields)

        return tree

    if isinstance(where, FunctionType) or isinstance(where, type(tree)):
        return _dataclass_unfilter(copy.copy(tree), where)
    raise TypeError("Where must be of same type as `tree`  or a `Callable` function")
