from __future__ import annotations

import dataclasses as dc
import functools as ft
from types import FunctionType
from typing import Any, Iterable

import jax.numpy as jnp
import jax.tree_util as jtu

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
