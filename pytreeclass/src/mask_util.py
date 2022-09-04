# this script defines some function that can be used to mask a pytree
# in essence it is a wrapper around jax.tree_util.tree_map
# the main difference is that dipsatch is used to define the mapping function
# for each type of the pytree, notably if the type is a jnp.ndarray
# then the mapping function jnp.{} is used other wise , standard python is used

import functools as ft
from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu

from pytreeclass.src.decorator_util import dispatch


@ft.partial(ft.partial, jtu.tree_map)
def is_inexact_array(node) -> bool:
    """Check if a node is a jnp.ndarray of inexact type (i.e. differentiable)."""
    return isinstance(node, jnp.ndarray) and jnp.issubdtype(node, jnp.inexact)


@ft.partial(ft.partial, jtu.tree_map)
def is_inexact(node) -> bool:
    """Check if a node is inexact type (i.e. differentiable)."""
    return is_inexact_array(node) or isinstance(node, (float, complex))


@ft.partial(ft.partial, jtu.tree_map)
def logical_not(node: Any) -> bool:
    """Element-wise logical not."""

    @dispatch(argnum=0)
    def _not(node):
        # apply `not` to non jnp.ndarray objects
        return not node

    @_not.register(jnp.ndarray)
    def _(node):
        # apply `jnp.logical_not` to jnp.ndarray objects
        return jnp.logical_not(node)

    return _not(node)


def logical_or(lhs, rhs, is_leaf=lambda x: x is None):
    """Element-wise logical or.

    Note:
        Beside using it to as `or` function, possible use case is to
        merge two pytrees of same structure by replacing the None nodes.

    Example:
        @pytc.treeclass
        class Foo:
            a: tuple[int]
        >>> logical_or(Foo(a=(1, None)), Foo(a=(None,2)))  # Foo(a=(1,2))
    """

    @dispatch(argnum=0)
    def or_func(x: Any, y: Any):
        # apply `or` to non jnp.ndarray objects
        return x or y

    @or_func.register(jnp.ndarray)
    @dispatch(argnum=1)
    def lhs_array(x: jnp.ndarray, y: Any):
        # lhs is jnp.ndarray and rhs is not
        return x or y

    @lhs_array.register(jnp.ndarray)
    def lhs_array_rhs_array(x: jnp.ndarray, y: jnp.ndarray):
        # lhs and rhs are both jnp.ndarray
        return jnp.logical_or(x, y)

    return jtu.tree_map(or_func, lhs, rhs, is_leaf=is_leaf)


def logical_and(lhs, rhs, is_leaf=None):
    """Element-wise logical and."""

    @dispatch(argnum=0)
    def and_func(x: Any, y: Any):
        # apply `and` to non jnp.ndarray objects
        return x and y

    @and_func.register(jnp.ndarray)
    @dispatch(argnum=1)
    def lhs_array(x: jnp.ndarray, y: Any):
        # lhs is jnp.ndarray and rhs is not
        return x and y

    @lhs_array.register(jnp.ndarray)
    def lhs_array_rhs_array(x: jnp.ndarray, y: jnp.ndarray):
        # lhs and rhs are both jnp.ndarray
        return jnp.logical_and(x, y)

    return jtu.tree_map(and_func, lhs, rhs, is_leaf=is_leaf)


def where(cond, x, y):
    return cond.at[cond].set(x).at[logical_not(cond)].set(y)


def logical_all(tree):
    """Element-wise logical all."""

    @dispatch(argnum=0)
    def _logical_all(tree):
        ...

    @_logical_all.register(bool)
    def _(tree):
        # apply `all` to non jnp.ndarray objects
        return tree is True

    @_logical_all.register(jnp.ndarray)
    def _(tree):
        # apply `jnp.logical_all` to jnp.ndarray objects
        return jnp.all(tree)

    return jtu.tree_reduce(
        lambda acc, cur: logical_and(acc, _logical_all(cur)), tree, initializer=True
    )
