from __future__ import annotations

import dataclasses
import functools as ft
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


def field(
    *, nondiff: bool = False, frozen: bool = False, **kwargs
) -> dataclasses.Field:
    """Similar to dataclasses.field but with additional arguments
    Args:
        nondiff: if True, the field will not be differentiated
        frozen: if True, the field will be frozen
        **kwargs: additional arguments to pass to dataclasses.field
    """
    metadata = kwargs.pop("metadata", {})
    metadata["nondiff"] = nondiff
    metadata["frozen"] = frozen
    metadata["static"] = nondiff or frozen
    return dataclasses.field(metadata=metadata, **kwargs)
