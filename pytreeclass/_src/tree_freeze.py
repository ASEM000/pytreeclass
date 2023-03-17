from __future__ import annotations

import copy
import hashlib
from contextlib import contextmanager
from typing import Any, Sequence

import jax
import jax.tree_util as jtu
import numpy as np

from pytreeclass._src.tree_decorator import _FIELD_MAP, _FROZEN, _VARS, _WRAPPED

PyTree = Any


def _hash_node(node: Any) -> int:
    if isinstance(node, (jax.Array, np.ndarray)):
        return int(hashlib.sha256(np.array(node).tobytes()).hexdigest(), 16)
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


def _unwrap(value: Any) -> Any:
    return value.unwrap() if isinstance(value, ImmutableWrapper) else value


class ImmutableWrapper:
    # base class for all immutable wrappers
    # that gets a special treatment inside `treeclass` wrapped classes
    # in essence, this wrapper is rendered transparent inside `treeclass` wrapped classes
    # so that the wrapped value can be accessed directly, without the need to call `unwrap`
    # this is useful for myriads of use cases, such as freezing a value to avoid updating it
    # by `jax` transformations, or wrapping a value to make it hashable.
    def __init__(self, x: Any) -> None:
        # disable composition of Wrappers
        getattr(self, _VARS)[_WRAPPED] = _unwrap(x)

    def unwrap(self) -> Any:
        return getattr(self, _WRAPPED)

    def __setattr__(self, _, __) -> None:
        raise AttributeError("Cannot assign to frozen instance.")

    def __delattr__(self, _: str) -> None:
        raise AttributeError("Cannot delete from frozen instance.")


def _tree_unwrap(value: PyTree) -> PyTree:
    # enables the transparent wrapper behavior iniside `treeclass` wrapped classes
    def is_leaf(x: Any) -> bool:
        return isinstance(x, ImmutableWrapper) or hasattr(x, _FIELD_MAP)

    return jtu.tree_map(_unwrap, value, is_leaf=is_leaf)


class _HashableWrapper(ImmutableWrapper):
    # used to wrap metadata to make it hashable
    # this is intended to wrap frozen values to avoid error when comparing
    # the metadata.
    def __eq__(self, rhs: Any) -> bool:
        if not isinstance(rhs, _HashableWrapper):
            return False
        return _hash_node(self.unwrap()) == _hash_node(rhs.unwrap())

    def __hash__(self) -> int:
        return _hash_node(self.unwrap())


class FrozenWrapper(ImmutableWrapper):
    def __repr__(self):
        return f"#{self.unwrap()!r}"

    def __eq__(self, rhs: Any) -> bool:
        if not isinstance(rhs, FrozenWrapper):
            return False
        return self.unwrap() == rhs.unwrap()

    def __hash__(self) -> int:
        return hash(self.unwrap())


def _frozen_flatten(tree: Any) -> tuple[tuple, Any]:
    return (None,), _HashableWrapper(tree.unwrap())


def _frozen_unflatten(treedef: Any, _: Sequence[Any]) -> PyTree:
    tree = object.__new__(FrozenWrapper)  # type: ignore
    getattr(tree, _VARS)[_WRAPPED] = treedef.unwrap()
    return tree


jtu.register_pytree_node(FrozenWrapper, _frozen_flatten, _frozen_unflatten)


def freeze(wrapped: Any) -> FrozenWrapper:
    r"""A wrapper to freeze a value to avoid updating it by `jax` transformations.

    Example:
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
    """Unfreeze `frozen` value, otherwise return the value itself.

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


def is_frozen(wrapped: Any) -> bool:
    """Returns True if the value is a frozen wrapper."""
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
    def mutate_step(tree: PyTree):
        if not hasattr(tree, _FIELD_MAP):
            return tree
        # shadow the class _FROZEN attribute with an
        # instance variable to temporarily disable the frozen behavior
        # after the context manager exits, the instance variable will be deleted
        # and the class attribute will be used again.
        getattr(tree, _VARS)[_FROZEN] = False
        for key in getattr(tree, _FIELD_MAP):
            mutate_step(getattr(tree, key))
        return tree

    def immutate_step(tree):
        if not hasattr(tree, _FIELD_MAP):
            return tree
        if _FROZEN not in getattr(tree, _VARS):
            return tree

        del getattr(tree, _VARS)[_FROZEN]
        for key in getattr(tree, _FIELD_MAP):
            immutate_step(getattr(tree, key))
        return tree

    tree = copy.copy(tree)
    mutate_step(tree)
    yield tree
    immutate_step(tree)
