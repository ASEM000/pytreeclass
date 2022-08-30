from __future__ import annotations

import functools as ft
import operator as op

import jax
import jax.tree_util as jtu

from pytreeclass.src.decorator_util import dispatch
from pytreeclass.src.tree_util import (
    is_treeclass,
    node_false,
    node_not,
    node_true,
    tree_copy,
)


def _dispatched_op_tree_map(func, lhs, rhs=None, is_leaf=None):
    """Slightly different implementation to jtu.tree_map for unary/binary operators broadcasting"""

    @dispatch(argnum=1)
    def _tree_map(lhs, rhs):
        raise NotImplementedError(f"rhs of type {type(rhs)} is not implemented.")

    @_tree_map.register(type(lhs))
    def _(lhs, rhs):
        return jtu.tree_map(func, lhs, rhs, is_leaf=is_leaf)

    @_tree_map.register(jax.interpreters.partial_eval.DynamicJaxprTracer)
    @_tree_map.register(int)
    @_tree_map.register(float)
    @_tree_map.register(complex)
    @_tree_map.register(bool)
    @_tree_map.register(str)
    def _(lhs, rhs):
        # broadcast the rhs to the lhs
        return jtu.tree_map(lambda x: func(x, rhs), lhs, is_leaf=is_leaf)

    @_tree_map.register(type(None))
    def _(lhs, rhs=None):
        return jtu.tree_map(func, lhs, is_leaf=is_leaf)

    return _tree_map(lhs, rhs)


def _dataclass_map(tree, cond, true_func=lambda x: x, false_func=lambda x: x):
    # we traverse the dataclass fields in a depth first manner
    # and apply true_func to field_value if condition is true and vice versa
    # unlike using jtu.tree_map which will traverse only the children of field_values
    def recurse(tree):
        for field_item in tree.__pytree_fields__.values():
            field_value = getattr(tree, field_item.name)

            if not field_item.metadata.get("static", False) and is_treeclass(
                field_value
            ):
                if cond(field_item, field_value):
                    object.__setattr__(
                        tree, field_item.name, jtu.tree_map(true_func, field_value)
                    )
                else:
                    recurse(field_value)

            else:
                object.__setattr__(
                    tree,
                    field_item.name,
                    true_func(field_value)
                    if cond(field_item, field_value)
                    else false_func(field_value),
                )

        return tree

    return recurse(tree_copy(tree))


def _append_math_op(func):
    """binary and unary magic operations"""

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
        def _(self, rhs):
            return _dispatched_op_tree_map(func, self, rhs)

        @inner_wrapper.register(str)
        def _(tree, where, **kwargs):
            """Filter by field name"""
            return _dataclass_map(
                tree,
                cond=lambda field_item, _: (field_item.name == where),
                true_func=node_true,
                false_func=node_false,
            )

        @inner_wrapper.register(type)
        def _(tree, where, **kwargs):
            """Filter by type"""
            return _dataclass_map(
                tree,
                cond=lambda _, field_value: isinstance(field_value, where),
                true_func=node_true,
                false_func=node_false,
            )

        @inner_wrapper.register(dict)
        def _(tree, where, **kwargs):
            """Filter by metadata"""

            def in_metadata(fld):
                kws, vals = zip(*where.items())
                return all(
                    fld.metadata.get(kw, False) and (fld.metadata[kw] == val)
                    for kw, val in zip(kws, vals)
                )

            return _dataclass_map(
                tree,
                cond=lambda field_item, _: in_metadata(field_item),
                true_func=node_true,
                false_func=node_false,
            )

        return (
            jtu.tree_map(node_not, inner_wrapper(self, rhs))
            if func == op.ne
            else inner_wrapper(self, rhs)
        )

    return wrapper


class _treeOpBase:

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
