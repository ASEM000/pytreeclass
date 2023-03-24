from __future__ import annotations

import copy
import hashlib
from contextlib import contextmanager
from typing import Any, Sequence

import jax
import jax.tree_util as jtu
import numpy as np

from pytreeclass._src.tree_decorator import _FROZEN, _WRAPPED, _field_registry, fields

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
    """Base class for all immutable wrappers that gets a special treatment inside `treeclass` wrapped classes.
    In essence, this wrapper is rendered transparent inside `treeclass` wrapped classes
    so that the wrapped value can be accessed directly, without the need to call `unwrap`.
    This behavior is useful for myriads of use cases, such as freezing a value to avoid updating it
    by `jax` transformations, or wrapping a value to make it hashable.

    Example:
        >>> import jax.tree_util as jut
        >>> import pytreeclass as pytc
        >>> @jtu.register_pytree_node_class
        ... class TransparentWrapper(pytc.ImmutableWrapper):
        ...    def __repr__(self):
        ...        return f"TransparentWrapper({self.__wrapped__!r})"
        ...    def tree_flatten(self):
        ...        # return the unwrapped value as a tuple
        ...        return (self.unwrap(),), None
        ...    @classmethod
        ...    def tree_unflatten(cls, _, xs):
        ...        return TransparentWrapper(xs[0])

        >>> @pytc.treeclass
        ... class Tree:
        ...    a:int = TransparentWrapper(1)
        >>> tree = Tree()
        >>> # no need to unwrap the value when accessing it
        >>> assert type(tree.a)  is int
        >>> print(tree)
        Tree(a=TransparentWrapper(1))
        >>> jtu.tree_leaves(tree)
        [1]
        >>> jtu.tree_leaves(tree, is_leaf=lambda x: isinstance(x, TransparentWrapper))
        [TransparentWrapper(1)]
    """

    def __init__(self, x: Any) -> None:
        # disable composition of Wrappers
        vars(self)[_WRAPPED] = _unwrap(x)

    def unwrap(self) -> Any:
        return getattr(self, _WRAPPED)

    def __setattr__(self, _, __) -> None:
        raise AttributeError("Cannot assign to frozen instance.")

    def __delattr__(self, _: str) -> None:
        raise AttributeError("Cannot delete from frozen instance.")


def _tree_unwrap(value: PyTree) -> PyTree:
    # enables the transparent wrapper behavior iniside `treeclass` wrapped classes
    def is_leaf(x: Any) -> bool:
        return isinstance(x, ImmutableWrapper) or type(x) in _field_registry

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
    vars(tree)[_WRAPPED] = treedef.unwrap()
    return tree


jtu.register_pytree_node(FrozenWrapper, _frozen_flatten, _frozen_unflatten)


def freeze(wrapped: Any) -> FrozenWrapper:
    r"""Freeze a value to avoid updating it by `jax` transformations.

    Example:
        >>> import jax
        >>> import pytreeclass as pytc
        >>> import jax.tree_util as jtu
        >>> # Usage with `jax.tree_util.tree_leaves`
        >>> # no leaves for a wrapped value
        >>> jtu.tree_leaves(pytc.freeze(2.))
        []

        >>> # retrieve the frozen wrapper value using `is_leaf=pytc.is_frozen`
        >>> jtu.tree_leaves(pytc.freeze(2.), is_leaf=pytc.is_frozen)
        [#2.0]

        >>> # Usage with `jax.tree_util.tree_map`
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
        >>> print(f"value: {value}\ngrad: {grad}")
        value: 4.0
        grad: Test(a=2.7725887)

        >>> # with `freeze` wrapping `a`, `a` will NOT be updated
        >>> value, grad = Test(a=pytc.freeze(2.))(2.)
        >>> print(f"value: {value}\ngrad: {grad}")
        value: 4.0
        grad: Test(a=#2.0)

        >>> # usage with `jax.tree_map` to freeze a tree
        >>> tree = Test(a = 2.)
        >>> frozen_tree = jax.tree_map(pytc.freeze, tree)
        >>> value, grad = frozen_tree(2.)
        >>> print(f"value: {value}\ngrad: {grad}")
        value: 4.0
        grad: Test(a=#2.0)
    """
    return FrozenWrapper(wrapped)


def unfreeze(x: Any) -> Any:
    """Unfreeze `frozen` value, otherwise return the value itself.

    - use `is_leaf=pytc.is_frozen` with `jax.tree_util.tree_map` to unfreeze a tree.**

    Example:
        >>> import pytreeclass as pytc
        >>> import jax.tree_util as jtu
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


def is_nondiff(x: Any) -> bool:
    """Returns False if the node is a float, complex number, or a numpy array of floats or complex numbers.

    Example:
        >>> import pytreeclass as pytc
        >>> import jax.numpy as jnp
        >>> pytc.is_nondiff(jnp.array(1))  # int array is non-diff type
        True
        >>> pytc.is_nondiff(jnp.array(1.))  # float array is diff type
        False
        >>> pytc.is_nondiff(1)  # int is non-diff type
        True
        >>> pytc.is_nondiff(1.)  # float is diff type
        False

    Note:
        This function is meant to be used with `jax.tree_util.tree_map` to create a mask
        for non-differentiable nodes in a tree, that can be used to freeze the non-differentiable nodes
        in a tree.
    """
    if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.inexact):
        return False
    if isinstance(x, (float, complex)):
        return False
    return True


@contextmanager
def _call_context(tree: PyTree):
    def mutate_step(tree: PyTree):
        if type(tree) not in _field_registry:
            return tree
        # shadow the class _FROZEN attribute with an
        # instance variable to temporarily disable the frozen behavior
        # after the context manager exits, the instance variable will be deleted
        # and the class attribute will be used again.
        vars(tree)[_FROZEN] = False  # type: ignore
        for field in fields(tree):
            mutate_step(getattr(tree, field.name))  # type: ignore
        return tree

    def immutate_step(tree):
        if type(tree) not in _field_registry:
            return tree
        if _FROZEN not in vars(tree):
            return tree

        del vars(tree)[_FROZEN]
        for field in fields(tree):
            immutate_step(getattr(tree, field.name))
        return tree

    tree = copy.copy(tree)
    mutate_step(tree)
    yield tree
    immutate_step(tree)
