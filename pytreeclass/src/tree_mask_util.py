import functools as ft
from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu

from pytreeclass.src.decorator_util import dispatch

# from pytreeclass.src.tree_util import static_field


@ft.partial(ft.partial, jtu.tree_map)
def is_inexact_array(node) -> bool:
    return isinstance(node, jnp.ndarray) and jnp.issubdtype(node, jnp.inexact)


@ft.partial(ft.partial, jtu.tree_map)
def is_inexact(node) -> bool:
    return is_inexact_array(node) or isinstance(node, (float, complex))


@ft.partial(ft.partial, jtu.tree_map)
def logical_not(node: Any) -> bool:
    @dispatch(argnum=0)
    def _not(node):
        return not node

    @_not.register(jnp.ndarray)
    def _(node):
        return jnp.logical_not(node)

    return _not(node)


@ft.partial(ft.partial, ft.partial(jtu.tree_map, is_leaf=lambda x: x is None))
def logical_or(lhs, rhs):
    @dispatch(argnum=0)
    def or_func(x: Any, y: Any):
        return x or y

    @or_func.register(jnp.ndarray)
    @dispatch(argnum=1)
    def lhs_array(x: jnp.ndarray, y: Any):
        return x or y

    @lhs_array.register(jnp.ndarray)
    def lhs_array_rhs_array(x: jnp.ndarray, y: jnp.ndarray):
        return jnp.logical_or(x, y)

    return or_func(lhs, rhs)


@ft.partial(ft.partial, ft.partial(jtu.tree_map, is_leaf=lambda x: x is None))
def logical_and(lhs, rhs):
    @dispatch(argnum=0)
    def and_func(x: Any, y: Any):
        return x and y

    @and_func.register(jnp.ndarray)
    @dispatch(argnum=1)
    def lhs_array(x: jnp.ndarray, y: Any):
        return x and y

    @lhs_array.register(jnp.ndarray)
    def lhs_array_rhs_array(x: jnp.ndarray, y: jnp.ndarray):
        return jnp.logical_and(x, y)

    return and_func(lhs, rhs)
