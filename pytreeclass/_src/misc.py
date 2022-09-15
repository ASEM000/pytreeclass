from __future__ import annotations

import functools as ft
from dataclasses import Field, field
from types import FunctionType
from typing import Any

from pytreeclass._src.tree_util import _tree_immutate, _tree_mutate

PyTree = Any


def _mutable(func):
    """decorator that allow mutable behvior
    for class methods/ function with treeclass as first argument

    Example:
        class ... :

        >>> @_mutable
        ... def mutable_method(self):
        ...    return self.value + 1
    """
    assert isinstance(
        func, FunctionType
    ), f"`mutable` can only be applied to methods. Found{type(func)}"

    @ft.wraps(func)
    def mutable_method(self, *args, **kwargs):
        self = _tree_mutate(tree=self)
        output = func(self, *args, **kwargs)
        self = _tree_immutate(tree=self)
        return output

    return mutable_method


class cached_method:
    def __init__(self, func):
        self.name = func.__name__
        self.func = func

    def __get__(self, instance, owner):
        output = self.func(instance)
        cached_func = ft.wraps(self.func)(lambda *args, **kwargs: output)
        object.__setattr__(instance, self.name, cached_func)
        return cached_func


def _field(name: str, type: type, metadata: dict[str, Any], repr: bool) -> Field:
    """field factory with option to add name, and type"""
    field_item = field(metadata=metadata, repr=repr)
    object.__setattr__(field_item, "name", name)
    object.__setattr__(field_item, "type", type)
    return field_item
