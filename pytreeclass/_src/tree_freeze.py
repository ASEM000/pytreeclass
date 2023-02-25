from __future__ import annotations

import copy
import dataclasses as dc
import operator as op
from contextlib import contextmanager
from typing import Any

import jax.tree_util as jtu
import numpy as np

from pytreeclass._src.tree_decorator import _FIELD_MAP, _FROZEN
from pytreeclass._src.tree_operator import _hash_node

PyTree = Any


class _Wrapper:
    def __init__(self, x: Any):
        # disable composition of Wrappers
        self.__wrapped__ = unfreeze(x)

    def unwrap(self):
        return self.__wrapped__

    def __setattr__(self, key, value):
        # allow setting the wrapped value only once.
        if "__wrapped__" in self.__dict__:
            raise dc.FrozenInstanceError("Cannot assign to frozen instance.")
        super().__setattr__(key, value)


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
    def __eq__(self, rhs: Any) -> bool:
        return op.eq(self.unwrap(), rhs)

    def __hash__(self) -> int:
        return hash(self.unwrap())

    def __repr__(self) -> str:
        return f"#{self.unwrap()!r}"

    def tree_flatten(self):
        return (None,), _HashableWrapper(self.unwrap())

    @classmethod
    def tree_unflatten(cls, treedef, _):
        self = object.__new__(cls)
        self.__dict__.update(__wrapped__=treedef.unwrap())
        return self


@contextmanager
def _call_context(tree: PyTree):
    def mutate_step(tree):
        if not hasattr(tree, _FIELD_MAP):
            return tree
        # shadow the class _FROZEN attribute with an
        # instance variable to temporarily disable the frozen behavior
        # after the context manager exits, the instance variable will be deleted
        # and the class attribute will be used again.
        tree.__dict__[_FROZEN] = False
        for key in getattr(tree, _FIELD_MAP):
            mutate_step(getattr(tree, key))
        return tree

    def immutate_step(tree):
        if not hasattr(tree, _FIELD_MAP):
            return tree
        if _FROZEN not in vars(tree):
            return tree

        del tree.__dict__[_FROZEN]
        for key in getattr(tree, _FIELD_MAP):
            immutate_step(getattr(tree, key))
        return tree

    tree = copy.copy(tree)
    mutate_step(tree)
    yield tree
    immutate_step(tree)


def freeze(x: Any) -> FrozenWrapper:
    """A wrapper to freeze a value inside a `treeclass` to avoid updating it by `jax` transformations.

    Example
        >>> @pytc.treeclass
        ... class Test:
        ...    a: float

        ... @jax.value_and_grad
        ... def __call__(self, x):
        ...    return x ** self.a

        >>> # without `freeze` wrapping `a`, `a` will be updated
        >>> value, grad = Test(a = 2.)(2.)
        >>> print("value:\t", value, "\ngrad:\t", grad)
        value:	 4.0
        grad:	 Test(a=2.7725887)

        >>> # with `freeze` wrapping `a`, `a` will NOT be updated
        >>> value, grad = Test(a=pytc.freeze(2.))(2.)
        >>> print("value:\t", value, "\ngrad:\t", grad)
        value:	 4.0
        grad:	 Test(a=#2.0)

        >>> # usage with `jax.tree_map` to freeze a tree
        >>> tree = Test(a = 2.)
        >>> frozen_tree = jax.tree_map(pytc.freeze, tree)
        >>> value, grad = frozen_tree(2.)
        >>> print("value:\t", value, "\ngrad:\t", grad)
        value:	 4.0
        grad:	 Test(a=#2.0)
    """
    return FrozenWrapper(x)


def unfreeze(x: Any) -> Any:
    """Unfreeze `frozen` value.

    - use `is_leaf=pytc.is_frozen` with `jax.tree_util.tree_map` to unfreeze a tree.**

    Example:
        >>> frozen_value = pytc.freeze(1)
        >>> pytc.unfreeze(frozen_value)
        1

        >>> # usage with `jax.tree_map`
        >>> frozen_tree = jtu.tree_map(pytc.freeze, {"a": 1, "b": 2})
        >>> unfrozen_tree = jtu.tree_map(pytc.unfreeze, frozen_tree, is_leaf=pytc.is_frozen)
        >>> unfrozen_tree
        {'a': 1, 'b': 2}
    """
    return x.unwrap() if isinstance(x, FrozenWrapper) else x


def is_frozen(node: Any) -> bool:
    """Check if a node is frozen"""
    return isinstance(node, FrozenWrapper)


def is_nondiff(node: Any) -> bool:
    """Returns False if the node is a float, complex number, or a numpy array of floats or complex numbers."""
    # this is meant to be used with `jtu.tree_map`.

    if hasattr(node, "dtype") and np.issubdtype(node.dtype, np.inexact):
        return False
    if isinstance(node, (float, complex)):
        return False
    return True
