from __future__ import annotations

import dataclasses
import functools as ft
from types import FunctionType
from typing import Any

import jax.tree_util as jtu

PyTree = Any


class _fieldDict(dict):
    """A dict used for `__treeclass_structure__` attribute of a treeclass instance"""

    # using a regular dict will cause the following error:
    # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def _tree_structure(tree) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return dynamic and static fields of the pytree instance"""
    # this function classifies tree vars into trainable/untrainable dict items
    # and returns a tuple of two dicts (dynamic, static)
    # that mark the tree leaves seen by JAX computations and the static(tree structure) that are
    # not seen by JAX computations. the scanning is done if the instance is not frozen.
    # otherwise the cached values are returned.
    static, dynamic = _fieldDict(tree.__dict__), _fieldDict()

    for field_item in dataclasses.fields(tree):
        if field_item.metadata.get("static", False) is False:
            dynamic[field_item.name] = static.pop(field_item.name)

    return (dynamic, static)


def tree_copy(tree):
    return jtu.tree_unflatten(*jtu.tree_flatten(tree)[::-1])


def _tree_mutate(tree):
    """Enable mutable behavior for a treeclass instance"""
    if dataclasses.is_dataclass(tree):
        tree.__dataclass_params__.frozen = False
        for field_item in dataclasses.fields(tree):
            if hasattr(tree, field_item.name):
                _tree_mutate(getattr(tree, field_item.name))
    return tree


def _tree_immutate(tree):
    """Enable immutable behavior for a treeclass instance"""
    if dataclasses.is_dataclass(tree):
        tree.__dataclass_params__.frozen = True
        for field_item in dataclasses.fields(tree):
            if hasattr(tree, field_item.name):
                _tree_immutate(getattr(tree, field_item.name))
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
        self = _tree_mutate(tree=self)
        output = func(self, *a, **k)
        self = _tree_immutate(tree=self)
        return output

    return mutable_method
