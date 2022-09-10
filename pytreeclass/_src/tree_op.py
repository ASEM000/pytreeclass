# this script is used to generate the magic methods for the tree classes
# the main idea is to use the jax.tree_map function to apply the operator to the tree
# possible lhs/rhs are scalar/jnp.ndarray or tree of the same type/structure

# Techincal note: the following code uses function dispatch heavily, to navigate
# through diffeent data types and how to handle each type.
# @dispatch is defined in dispatch.py and is based on functools.singledispatch

from __future__ import annotations

import functools as ft
import operator as op
from typing import Any

import jax
import jax.tree_util as jtu

from pytreeclass._src.dispatch import dispatch
from pytreeclass._src.tree_util import (
    _node_false,
    _node_not,
    _node_true,
    _pytree_map,
    _tree_hash,
)

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

    @_tree_map.register(jax.interpreters.partial_eval.DynamicJaxprTracer)
    @_tree_map.register(jax.numpy.ndarray)
    @_tree_map.register(int)
    @_tree_map.register(float)
    @_tree_map.register(complex)
    @_tree_map.register(bool)
    @_tree_map.register(str)
    def _(
        lhs,
        rhs: int
        | float
        | complex
        | bool
        | str
        | jax.numpy.ndarray
        | jax.interpreters.partial_eval.DynamicJaxprTracer,
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


def _append_math_eq_ne(func):
    """Append eq/ne operations"""
    assert func in [op.eq, op.ne], f"func={func} is not implemented."

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
        @inner_wrapper.register(jax.interpreters.partial_eval.DynamicJaxprTracer)
        @inner_wrapper.register(jax.numpy.ndarray)
        def _(
            self,
            rhs: int
            | float
            | complex
            | bool
            | type(self)
            | jax.interpreters.partial_eval.DynamicJaxprTracer
            | jax.numpy.ndarray,
        ):
            # this function is handling all the numeric types
            return _dispatched_op_tree_map(func, self, rhs)

        @inner_wrapper.register(str)
        def _(tree, where: str, **kwargs):
            """Filter by field name"""
            # here _pytree_map is used to traverse the tree depth first
            # and broadcast True/False to the children values
            # if the field name matches the where `string``
            return _pytree_map(
                tree,
                cond=lambda _, field_item, __: (field_item.name == where),
                true_func=lambda _, __, node_item: _node_true(node_item),
                false_func=lambda _, __, node_item: _node_false(node_item),
                attr_func=lambda _, field_item, __: field_item.name,
                is_leaf=lambda _, field_item, __: field_item.metadata.get("static", False),  # fmt: skip
            )

        @inner_wrapper.register(type)
        def _(tree, where: type, **kwargs):
            """Filter by field type"""
            # here _pytree_map is used to traverse the tree depth first
            # and broadcast True/False to the children values
            # if the field type matches the where `type`
            return _pytree_map(
                tree,
                # condition to check for each node
                cond=lambda _, __, node_item: isinstance(node_item, where),
                # if the condition is True, then broadcast True to the children
                true_func=lambda _, __, node_item: _node_true(node_item),
                # if the condition is False, then broadcast False to the children
                false_func=lambda _, __, node_item: _node_false(node_item),
                # which attribute to use in the object.__setattr__ function
                attr_func=lambda _, field_item, __: field_item.name,
                # if the node is a leaf, then do not traverse the children
                is_leaf=lambda _, field_item, __: field_item.metadata.get("static", False),  # fmt: skip
            )

        @inner_wrapper.register(dict)
        def _(tree, where: dict, **kwargs):
            """Filter by metadata"""
            # here _pytree_map is used to traverse the tree depth first
            # and broadcast True/False to the children values
            # if the field metadata contains the where `dict`
            # this mechanism could filter by multiple metadata, however possible drawbacks
            # are that some data structures might have the same metadata for all of it's elements (list/dict/tuple)
            # and this would filter out all the elements without distinction
            return _pytree_map(
                tree,
                # condition to check for each dataclass field
                cond=lambda _, field_item, __: (where == field_item.metadata),
                # if the condition is True, then broadcast True to the children
                true_func=lambda _, __, node_item: _node_true(node_item),
                # if the condition is False, then broadcast False to the children
                false_func=lambda _, __, node_item: _node_false(node_item),
                # which attribute to use in the object.__setattr__ function
                attr_func=lambda _, field_item, __: field_item.name,
                # if the node is a leaf, then do not traverse the children
                is_leaf=lambda _, field_item, __: field_item.metadata.get("static", False),  # fmt: skip
            )

        return (
            jtu.tree_map(_node_not, inner_wrapper(self, rhs))
            if func == op.ne
            else inner_wrapper(self, rhs)
        )

    return wrapper


class _treeOp:

    __hash__ = _tree_hash
    __abs__ = _append_math_op(op.abs)
    __add__ = _append_math_op(op.add)
    __radd__ = _append_math_op(op.add)
    __and__ = _append_math_op(op.and_)
    __rand__ = _append_math_op(op.and_)
    __eq__ = _append_math_eq_ne(op.eq)
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
    __ne__ = _append_math_eq_ne(op.ne)
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
