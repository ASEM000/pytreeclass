from __future__ import annotations

import dataclasses as dc
import functools as ft
from types import FunctionType
from typing import Any

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


def field(
    *,
    nondiff: bool = False,
    default=dc.MISSING,
    default_factory=dc.MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
    kw_only=dc.MISSING,
):
    """dataclass field with additional `nondiff` flag"""

    if default is not dc.MISSING and default_factory is not dc.MISSING:
        raise ValueError("cannot specify both default and default_factory")

    args = dict(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )

    if "kw_only" in dir(dc.Field):
        args.update(kw_only=kw_only)

    if nondiff:
        return pytc.NonDiffField(**args)

    return dc.Field(**args)
