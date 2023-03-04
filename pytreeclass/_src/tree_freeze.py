from __future__ import annotations

import copy
import functools as ft
import hashlib
from contextlib import contextmanager
from typing import Any

import jax.tree_util as jtu
import numpy as np

from pytreeclass._src.tree_decorator import _FIELD_MAP, _FROZEN, _WRAPPED

PyTree = Any


def _hash_node(node: PyTree) -> int:
    if hasattr(node, "dtype") and hasattr(node, "shape"):
        return hashlib.sha256(np.array(node).tobytes()).hexdigest()
    if isinstance(node, set):
        return hash(frozenset(node))
    if isinstance(node, dict):
        return hash(frozenset(node.items()))
    if isinstance(node, list):
        return hash(tuple(node))
    return hash(node)


def _tree_hash(tree: PyTree) -> int:
    hashed = jtu.tree_map(_hash_node, jtu.tree_leaves(tree))
    return hash((*hashed, jtu.tree_structure(tree)))


class ImmutableWrapper:
    def __init__(self, x: Any):
        # disable composition of Wrappers
        vars(self)[_WRAPPED] = x.unwrap() if isinstance(x, ImmutableWrapper) else x

    def unwrap(self):
        return getattr(self, _WRAPPED)

    def __setattr__(self, key, value):
        # allow setting the wrapped value only once.
        if _WRAPPED in vars(self):
            raise AttributeError("Cannot assign to frozen instance.")
        super().__setattr__(key, value)

    def __delattr__(self, __name: str) -> None:
        raise AttributeError("Cannot delete from frozen instance.")


def _unwrap(wrapped: Any) -> Any:
    if isinstance(wrapped, ImmutableWrapper):
        return wrapped.unwrap()
    return wrapped


class _HashableWrapper(ImmutableWrapper):
    # used to wrap metadata to make it hashable
    # this is intended to wrap frozen values to avoid error when comparing
    # the metadata.
    def __eq__(self, rhs: Any) -> bool:
        if not isinstance(rhs, _HashableWrapper):
            return False
        return _hash_node(self.unwrap()) == _hash_node(rhs.unwrap())

    def __hash__(self):
        return _hash_node(self.unwrap())


class FrozenWrapper(ImmutableWrapper):
    def __getattr__(self, key):
        # delegate non magical attributes to the wrapped value
        return getattr(self.unwrap(), key)

    def __repr__(self):
        return f"#{self.unwrap()!r}"

    def __eq__(self, rhs: Any) -> bool:
        if not isinstance(rhs, FrozenWrapper):
            return False
        return self.unwrap() == rhs.unwrap()

    def __hash__(self) -> int:
        return hash(self.unwrap())


def _flatten(tree):
    return (None,), _HashableWrapper(tree.unwrap())


def _unflatten(klass, treedef, _):
    tree = object.__new__(klass)
    vars(tree)[_WRAPPED] = treedef.unwrap()
    return tree


jtu.register_pytree_node(FrozenWrapper, _flatten, ft.partial(_unflatten, FrozenWrapper))


def freeze(wrapped: Any) -> FrozenWrapper:
    """A wrapper to freeze a value to avoid updating it by `jax` transformations.

    Example
        >>> import jax
        >>> import pytreeclass as pytc
        >>> import jax.tree_util as jtu

        ** usage with `jax.tree_util.tree_leaves` **
        >>> # no leaves for a wrapped value
        >>> jtu.tree_leaves(pytc.freeze(2.))
        []

        >>> # retrieve the frozen wrapper value using `is_leaf=pytc.is_frozen`
        >>> jtu.tree_leaves(pytc.freeze(2.), is_leaf=pytc.is_frozen)
        [#2.0]

        ** usage with `jax.tree_util.tree_map` **
        >>> a= [1,2,3]
        >>> a[1] = pytc.freeze(a[1])
        >>> jtu.tree_map(lambda x:x+100, a)
        [101, #2, 103]

        >>> @pytc.treeclass
        ... class Test:
        ...     a: float
        ...     @jax.value_and_grad
        ...     def __call__(self, x):
        ...         return x ** self.a

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
    return FrozenWrapper(wrapped)


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


def is_frozen(wrapped):
    return isinstance(wrapped, FrozenWrapper)


def is_nondiff(node: Any) -> bool:
    """Returns False if the node is a float, complex number, or a numpy array of floats or complex numbers."""
    # this is meant to be used with `jtu.tree_map`.

    if hasattr(node, "dtype") and np.issubdtype(node.dtype, np.inexact):
        return False
    if isinstance(node, (float, complex)):
        return False
    return True


@contextmanager
def _call_context(tree: PyTree):
    def mutate_step(tree):
        if not hasattr(tree, _FIELD_MAP):
            return tree
        # shadow the class _FROZEN attribute with an
        # instance variable to temporarily disable the frozen behavior
        # after the context manager exits, the instance variable will be deleted
        # and the class attribute will be used again.
        vars(tree)[_FROZEN] = False
        for key in getattr(tree, _FIELD_MAP):
            mutate_step(getattr(tree, key))
        return tree

    def immutate_step(tree):
        if not hasattr(tree, _FIELD_MAP):
            return tree
        if _FROZEN not in vars(tree):
            return tree

        del vars(tree)[_FROZEN]
        for key in getattr(tree, _FIELD_MAP):
            immutate_step(getattr(tree, key))
        return tree

    tree = copy.copy(tree)
    mutate_step(tree)
    yield tree
    immutate_step(tree)
