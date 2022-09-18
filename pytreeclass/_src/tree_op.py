# this script is used to generate the magic methods for the tree classes
# the main idea is to use the jax.tree_map function to apply the operator to the tree
# possible lhs/rhs are scalar/jnp.ndarray or tree of the same type/structure

# Techincal note: the following code uses function dispatch heavily, to navigate
# through diffeent data types and how to handle each type.
# @dispatch is defined in dispatch.py and is based on functools.singledispatch

from __future__ import annotations

import functools as ft
import operator as op
from collections.abc import Iterable
from dataclasses import Field
from typing import Any, Callable, Generator

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax._src.tree_util import flatten_one_level
from jax.core import Tracer

from pytreeclass._src.dispatch import dispatch
from pytreeclass._src.tree_util import _tree_fields, is_treeclass

PyTree = Any


def _dispatched_op_tree_map(func, lhs, rhs=None, is_leaf=None):
    """`jtu.tree_map` for unary/binary operators broadcasting"""

    @dispatch(argnum=1)
    def _tree_map(lhs, rhs):
        raise NotImplementedError(f"rhs of type {type(rhs)} is not implemented.")

    @_tree_map.register(type(lhs))
    def _(lhs, rhs):
        # if rhs is a tree, then apply the operator to the tree
        # the rhs tree here must be of the same type as lhs tree
        return jtu.tree_map(func, lhs, rhs, is_leaf=is_leaf)

    @_tree_map.register(Tracer)
    @_tree_map.register(jnp.ndarray)
    @_tree_map.register(int)
    @_tree_map.register(float)
    @_tree_map.register(complex)
    @_tree_map.register(bool)
    @_tree_map.register(str)
    def _(
        lhs,
        rhs: int | float | complex | bool | str | jnp.ndarray | Tracer,
    ):
        # broadcast the scalar rhs to the lhs
        return jtu.tree_map(lambda x: func(x, rhs), lhs, is_leaf=is_leaf)

    @_tree_map.register(type(None))
    def _(lhs, rhs=None):
        # if rhs is None, then apply the operator to the tree
        # i.e. this defines the unary operator
        return jtu.tree_map(func, lhs, is_leaf=is_leaf)

    return _tree_map(lhs, rhs)


def _append_math_op(func):
    """binary and unary magic operations"""
    # make `func` work on pytree

    @ft.wraps(func)
    def wrapper(self, rhs=None):
        return _dispatched_op_tree_map(func, self, rhs)

    return wrapper


def _field_boolean_map(
    cond: Callable[[Field, Any], bool],
    tree: PyTree,
) -> PyTree:
    """Set node True if cond(field, value) is True, otherwise set node False

    Args:
        cond (Callable[[Field, Any], bool]): Condition function applied to each field
        tree (PyTree): _description_
        is_leaf (Callable[[Any], bool] | None, optional): is_leaf. Defaults to None.

    Returns:
        PyTree: boolean mapped tree
    """
    # this is the function responsible for the boolean mapping of
    # `node_type`, `field_name`, and `field_metadata` comparisons.

    def _node_true(node):
        if isinstance(node, jnp.ndarray):
            # broadcast True to the shape of the array
            return jnp.ones_like(node).astype(jnp.bool_)
        elif isinstance(node, Iterable):
            # broadcast False to the node
            return jtu.tree_map(lambda _: True, node)
        else:
            return True

    def _node_false(node):
        if isinstance(node, jnp.ndarray):
            # broadcast False to the shape of the array
            return jnp.zeros_like(node).astype(jnp.bool_)
        elif isinstance(node, Iterable):
            # broadcast False to the node
            return jtu.tree_map(lambda _: False, node)
        else:
            return False

    def _is_leaf(node):
        return isinstance(node, Iterable) or node is None

    def _traverse(tree) -> Generator[Any, ...]:
        """traverse the tree and yield the applied function on the field and node"""
        # We check each level of the tree not tree leaves,
        # this is because, if a condition is met at a parent tree
        # then the entire subtree is marked by a `True` subtree of the same structure.
        # for example let `Test` be a pytreeclass wrapped class
        # >>> tree = Test(a=1, b=Test(c=2,d=3))
        # if we  check a field name == "b", then the entire subtree at b is marked True
        # however if we get the tree_leaves of the tree, `b` will not be visible to the condition.

        leaves = flatten_one_level(tree)[0]
        field_items = _tree_fields(tree).values()

        for field_item, node_item in zip(field_items, leaves):
            condition = cond(field_item, node_item)

            if is_treeclass(node_item):
                yield from [
                    _node_true(item)
                    for item in jtu.tree_leaves(node_item, is_leaf=_is_leaf)
                ] if condition else _traverse(tree=node_item)

            else:
                yield _node_true(node_item) if condition else _node_false(node_item)

    return jtu.tree_unflatten(
        treedef=jtu.tree_structure(tree, is_leaf=_is_leaf),
        leaves=tuple(_traverse(tree=tree)),
    )


def _append_math_eq_ne(func):
    """Append eq/ne operations"""

    @ft.wraps(func)
    def wrapper(self, rhs):
        @dispatch(argnum=1)
        def inner_wrapper(tree, where, **kwargs):
            raise NotImplementedError(f"rhs of type {type(rhs)} is not implemented.")

        @inner_wrapper.register(int)
        @inner_wrapper.register(float)
        @inner_wrapper.register(complex)
        @inner_wrapper.register(bool)
        @inner_wrapper.register(type(self))
        @inner_wrapper.register(Tracer)
        @inner_wrapper.register(jnp.ndarray)
        def _(
            self,
            rhs: int | float | complex | bool | type(self) | Tracer | jnp.ndarray,
        ):
            # this function is handling all the numeric types
            return _dispatched_op_tree_map(func, self, rhs)

        @inner_wrapper.register(str)
        def _(tree, where: str, **kwargs):
            """Filter by field name"""
            return _field_boolean_map(lambda x, y: func(x.name, where), tree)

        @inner_wrapper.register(type)
        def _(tree, where: type, **kwargs):
            """Filter by field type"""
            return _field_boolean_map(lambda x, y: func(y, where), tree)

        @inner_wrapper.register(dict)
        def _(tree, where: dict[str, Any], **kwargs):
            """Filter by metadata"""
            return _field_boolean_map(lambda x, y: func(x.metadata, where), tree)

        return inner_wrapper(self, rhs)

    return wrapper


def _tree_hash(tree):
    """Return a hash of the tree"""

    def _hash_node(node):
        """hash the leaves of the tree"""
        if isinstance(node, jnp.ndarray):
            return np.array(node).tobytes()
        elif isinstance(node, set):
            # jtu.tree_map does not traverse sets
            return frozenset(node)
        else:
            return node

    return hash(
        (*jtu.tree_map(_hash_node, jtu.tree_leaves(tree)), jtu.tree_structure(tree))
    )


def _eq(lhs, rhs):
    # (x == int)  <-> isinstance(x,int)
    return isinstance(lhs, rhs) if isinstance(rhs, type) else op.eq(lhs, rhs)


def _ne(lhs, rhs):
    return not isinstance(lhs, rhs) if isinstance(rhs, type) else op.ne(lhs, rhs)


class _treeOp:

    __hash__ = _tree_hash
    __abs__ = _append_math_op(op.abs)
    __add__ = _append_math_op(op.add)
    __radd__ = _append_math_op(op.add)
    __and__ = _append_math_op(op.and_)
    __rand__ = _append_math_op(op.and_)
    __eq__ = _append_math_eq_ne(_eq)
    __floordiv__ = _append_math_op(op.floordiv)
    __ge__ = _append_math_op(op.ge)
    __gt__ = _append_math_op(op.gt)
    __inv__ = _append_math_op(op.inv)
    __invert__ = _append_math_op(op.invert)
    __le__ = _append_math_op(op.le)
    __lshift__ = _append_math_op(op.lshift)
    __lt__ = _append_math_op(op.lt)
    __matmul__ = _append_math_op(op.matmul)
    __mod__ = _append_math_op(op.mod)
    __mul__ = _append_math_op(op.mul)
    __rmul__ = _append_math_op(op.mul)
    __ne__ = _append_math_eq_ne(_ne)
    __neg__ = _append_math_op(op.neg)
    __not__ = _append_math_op(op.not_)
    __or__ = _append_math_op(op.or_)
    __pos__ = _append_math_op(op.pos)
    __pow__ = _append_math_op(op.pow)
    __rshift__ = _append_math_op(op.rshift)
    __sub__ = _append_math_op(op.sub)
    __rsub__ = _append_math_op(op.sub)
    __truediv__ = _append_math_op(op.truediv)
    __xor__ = _append_math_op(op.xor)
