from __future__ import annotations

from typing import Any, Iterable

import jax.tree_util as jtu
import numpy as np

from pytreeclass._src.tree_operator import _hash_node

PyTree = Any


class _Wrapper:
    def __init__(self, x: Any):
        # disable composition of Wrappers
        self.__wrapped__ = x.unwrap() if isinstance(x, _Wrapper) else x

    def unwrap(self):
        return self.__wrapped__


class _HashableWrapper(_Wrapper):
    # used to wrap metadata to make it hashable
    # this is intended to wrap frozen values to avoid error when comparing
    # the metadata.
    def __eq__(self, rhs: Any) -> bool:
        if not isinstance(rhs, _HashableWrapper):
            return False
        return _hash_node(self.unwrap()) == _hash_node(rhs.unwrap())

    def __hash__(self):
        return _hash_node(self.unwrap())


@jtu.register_pytree_node_class
class FrozenWrapper(_Wrapper):
    def __setattr__(self, key: str, value: Any) -> None:
        if "__wrapped__" in self.__dict__:
            msg = "FrozenWrapper only allows `__wrapped__` to be set once`"
            raise ValueError(msg)
        return super().__setattr__(key, value)

    def __getattr__(self, k):
        return getattr(self.unwrap(), k)

    def __call__(self, *a, **k):
        return self.unwrap()(*a, **k)

    def tree_flatten(self):
        # Wrapping the metadata to ensure its hashability and equality
        # https://github.com/google/jax/issues/13027
        return (None,), _HashableWrapper(self.unwrap())

    @classmethod
    def tree_unflatten(cls, treedef, leaves):
        self = object.__new__(cls)
        self.__dict__.update(__wrapped__=treedef.unwrap())
        return self

    def __repr__(self):
        return f"#{self.unwrap()!r}"

    def __eq__(self, rhs: Any) -> bool:
        return (self.unwrap() == rhs.unwrap()) if is_frozen(rhs) else False

    def __hash__(self):
        return hash(self.unwrap())


def is_frozen(node: Any) -> bool:
    """Check if a tree is wrapped by a wrapper"""
    return isinstance(node, FrozenWrapper)


def tree_freeze(x: PyTree) -> PyTree:
    """Freeze tree leaf"""
    return jtu.tree_map(FrozenWrapper, x)


def tree_unfreeze(x: PyTree) -> PyTree:
    """Unfreeze tree leaf"""
    # this is a bit tricky as we are using `is_leaf` to stop
    # traversing the tree when we hit a `FrozenWrapper`
    # the problem here is that, unlike `tree_freeze` this function
    # can not be used inside a `jtu.tree_map` **without** specifying
    # `is_leaf` as it will traverse the whole tree and miss the wrapper mark
    map_func = lambda x: x.unwrap() if is_frozen(x) else x
    return jtu.tree_map(map_func, x, is_leaf=is_frozen)


def is_nondiff(item: Any) -> bool:
    """Check if a node is non-differentiable."""

    def _is_nondiff_item(node: Any):
        if hasattr(node, "dtype") and np.issubdtype(node.dtype, np.inexact):
            return False
        if isinstance(node, (float, complex)):
            return False
        return True

    if isinstance(item, Iterable):
        # if an iterable has at least one non-differentiable item
        # then the whole iterable is non-differentiable
        return any([_is_nondiff_item(item) for item in jtu.tree_leaves(item)])

    return _is_nondiff_item(item)
