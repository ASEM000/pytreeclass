# this script is used to generate the magic methods for the tree classes
# the main idea is to use the jax.tree_map function to apply the operator to the tree
# possible lhs/rhs are scalar/jnp.ndarray or tree of the same type/structure

from __future__ import annotations

import functools as ft
import operator as op
from dataclasses import Field
from typing import Any, Callable, Generator

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.core import Tracer

import pytreeclass as pytc
from pytreeclass._src.tree_util import tree_copy

PyTree = Any


def _dispatched_op_tree_map(func, lhs, rhs=None, is_leaf=None):
    """`jtu.tree_map` for unary/binary operators broadcasting"""
    if isinstance(rhs, type(lhs)):
        return jtu.tree_map(func, lhs, rhs, is_leaf=is_leaf)
    elif isinstance(rhs, (Tracer, jnp.ndarray, int, float, complex, bool, str)):
        return jtu.tree_map(lambda x: func(x, rhs), lhs, is_leaf=is_leaf)
    elif isinstance(rhs, type(None)):  # unary operator
        return jtu.tree_map(func, lhs, is_leaf=is_leaf)
    else:
        raise NotImplementedError(f"rhs of type {type(rhs)} is not implemented.")


def _append_math_op(func):
    """binary and unary magic operations"""

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

    def _true_leaves(node: Any) -> list[bool, ...]:
        return [
            jnp.ones_like(leaf).astype(jnp.bool_)
            if isinstance(leaf, jnp.ndarray)
            else True
            for leaf in jtu.tree_leaves(node, is_leaf=_is_leaf)
        ]

    def _false_leaves(node: Any) -> list[bool, ...]:
        return [
            jnp.zeros_like(leaf).astype(jnp.bool_)
            if isinstance(leaf, jnp.ndarray)
            else False
            for leaf in jtu.tree_leaves(node, is_leaf=_is_leaf)
        ]

    def _is_leaf(node: Any) -> bool:
        return node is None

    def _traverse(tree) -> Generator[Any, ...]:
        """traverse the tree and yield the applied function on the field and node"""
        # We check each level of the tree not tree leaves,
        # this is because, if a condition is met at a parent tree
        # then the entire subtree is marked by a `True` subtree of the same structure.
        # for example let `Test` be a pytreeclass wrapped class
        # >>> tree = Test(a=1, b=Test(c=2,d=3))
        # if we  check a field name == "b", then the entire subtree at b is marked True
        # however if we get the tree_leaves of the tree, `b` will not be visible to the condition.

        for field_item, node_item in (
            [f, getattr(tree, f.name)]
            for f in pytc.fields(tree)
            if not f.metadata.get("static", False)
        ):

            yield from _true_leaves(node_item) if cond(field_item, node_item) else (
                _traverse(node_item)
                if pytc.is_treeclass(node_item)
                else _false_leaves(node_item)
            )

    return jtu.tree_unflatten(
        treedef=jtu.tree_structure(tree, is_leaf=_is_leaf),
        leaves=tuple(_traverse(tree=tree)),
    )


def _append_math_eq_ne(func):
    """Append eq/ne operations"""

    @ft.wraps(func)
    def wrapper(self, where):

        if isinstance(where, (int, float, complex, bool, type(self), Tracer, jnp.ndarray)):  # fmt: skip
            return _dispatched_op_tree_map(func, self, where)
        elif isinstance(where, str):
            return _field_boolean_map(lambda x, y: func(x.name, where), self)
        elif isinstance(where, type):
            return _field_boolean_map(lambda x, y: func(y, where), self)
        elif isinstance(where, dict):
            return _field_boolean_map(lambda x, y: func(x.metadata, where), self)
        else:
            raise NotImplementedError(f"rhs of type {type(where)} is not implemented.")

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
    __copy__ = tree_copy
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
