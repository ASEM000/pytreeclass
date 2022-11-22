# this script is used to generate the magic methods for the tree classes
# the main idea is to use the jax.tree_map function to apply the operator to the tree
# possible lhs/rhs are scalar/jnp.ndarray or tree of the same type/structure

from __future__ import annotations

import dataclasses as dc
import functools as ft
import operator as op
import re
from typing import Any, Callable, Generator

import jax.numpy as jnp
import jax.tree_util as jtu
from jax.core import Tracer

PyTree = Any


def _dispatched_op_tree_map(func, lhs, rhs=None, is_leaf=None):
    """`jtu.tree_map` for unary/binary operators broadcasting"""
    if isinstance(rhs, type(lhs)):
        return jtu.tree_map(func, lhs, rhs, is_leaf=is_leaf)
    elif isinstance(rhs, (Tracer, jnp.ndarray, int, float, complex, bool, str)):
        return jtu.tree_map(lambda x: func(x, rhs), lhs, is_leaf=is_leaf)
    elif isinstance(rhs, type(None)):  # unary operator
        return jtu.tree_map(func, lhs, is_leaf=is_leaf)
    raise NotImplementedError(f"rhs of type {type(rhs)} is not implemented.")


def _append_math_op(func):
    """binary and unary magic operations"""

    @ft.wraps(func)
    def wrapper(self, rhs=None):
        return _dispatched_op_tree_map(func, self, rhs)

    return wrapper


def _true_leaves(node: Any) -> list[bool, ...]:
    return [
        jnp.ones_like(leaf).astype(jnp.bool_) if isinstance(leaf, jnp.ndarray) else True
        for leaf in jtu.tree_leaves(node, is_leaf=lambda x: x is None)
    ]


def _false_leaves(node: Any) -> list[bool, ...]:
    return [
        jnp.zeros_like(leaf).astype(jnp.bool_)
        if isinstance(leaf, jnp.ndarray)
        else False
        for leaf in jtu.tree_leaves(node, is_leaf=lambda x: x is None)
    ]


def _field_boolean_map(cond: Callable[[dc.Field, Any], bool], tree: PyTree) -> PyTree:
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
            for f in dc.fields(tree)
            if not f.metadata.get("static", False)
        ):

            yield from _true_leaves(node_item) if cond(field_item, node_item) else (
                _traverse(node_item)
                if dc.is_dataclass(node_item)
                else _false_leaves(node_item)
            )

    return jtu.tree_unflatten(
        treedef=jtu.tree_structure(tree, is_leaf=lambda x: x is None),
        leaves=_traverse(tree=tree),
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
        raise NotImplementedError(f"rhs of type {type(where)} is not implemented.")

    return wrapper


def _eq(lhs, rhs):
    if isinstance(rhs, type):
        # rhs is a type (tree == int ) will perform a instance check
        return isinstance(lhs, rhs)
    elif isinstance(rhs, str):
        # rhs is a string for (tree == "a") will perform a field name check using regex
        found = re.findall(rhs, lhs)
        return len(found) > 0 and found[0] == lhs
    return op.eq(lhs, rhs)


def _ne(lhs, rhs):
    if isinstance(rhs, type):
        # rhs is a type (tree == int ) will perform a instance check
        return not isinstance(lhs, rhs)
    elif isinstance(rhs, str):
        # rhs is a string for (tree == "a") will perform a field name check using regex
        found = re.findall(rhs, lhs)
        return len(found) == 0 or found[0] != lhs
    return op.ne(lhs, rhs)
