from __future__ import annotations

import functools as ft
import operator as op

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from pytreeclass.src.decorator_util import dispatch
from pytreeclass.src.tree_util import is_excluded, is_treeclass


def _dispatched_op_tree_map(func, lhs, rhs=None, is_leaf=None):
    """Slightly different implementation to jtu.tree_map for unary/binary operators broadcasting"""

    @dispatch(argnum=1)
    def _tree_map(lhs, rhs):
        raise NotImplementedError(f"rhs of type {type(rhs)} is not implemented.")

    @_tree_map.register(type(lhs))
    def _(lhs, rhs):
        lhs_leaves, lhs_treedef = jtu.tree_flatten(lhs, is_leaf=is_leaf)
        rhs_leaves, rhs_treedef = jtu.tree_flatten(rhs, is_leaf=is_leaf)

        lhs_leaves = [
            func(lhs_leaf, rhs_leaf) if rhs_leaf is not None else lhs_leaf
            for (lhs_leaf, rhs_leaf) in zip(lhs_leaves, rhs_leaves)
        ]

        return jtu.tree_unflatten(lhs_treedef, lhs_leaves)

    @_tree_map.register(jax.interpreters.partial_eval.DynamicJaxprTracer)
    @_tree_map.register(int)
    @_tree_map.register(float)
    @_tree_map.register(complex)
    @_tree_map.register(bool)
    @_tree_map.register(str)
    def _(lhs, rhs):
        lhs_leaves, lhs_treedef = jtu.tree_flatten(lhs, is_leaf=is_leaf)
        lhs_leaves = [func(leaf, rhs) for leaf in lhs_leaves]
        return jtu.tree_unflatten(lhs_treedef, lhs_leaves)

    @_tree_map.register(type(None))
    def _(lhs, rhs=None):
        lhs_leaves, lhs_treedef = jtu.tree_flatten(lhs, is_leaf=is_leaf)
        lhs_leaves = [func(lhs_node) for lhs_node in lhs_leaves]
        return jtu.tree_unflatten(lhs_treedef, lhs_leaves)

    return _tree_map(lhs, rhs)


def _append_math_op(func):
    """binary and unary magic operations"""

    @ft.wraps(func)
    def wrapper(self, rhs=None):
        return _dispatched_op_tree_map(func, self, rhs)

    return wrapper


def _append_math_eq_ne(func):
    """Append eq/ne operations"""
    assert func in [op.eq, op.ne], f"func={func} is not implemented."

    def node_not(node):
        if isinstance(node, jnp.ndarray):
            return jnp.logical_not(node)
        else:
            return not node

    def set_true(node, array_as_leaves: bool = True):
        if isinstance(node, jnp.ndarray):
            return jnp.ones_like(node).astype(jnp.bool_) if array_as_leaves else True
        else:
            return True

    def set_false(node, array_as_leaves: bool = True):
        if isinstance(node, jnp.ndarray):
            return jnp.zeros_like(node).astype(jnp.bool_) if array_as_leaves else False
        else:
            return False

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
            tree_copy = jtu.tree_unflatten(*jtu.tree_flatten(tree)[::-1])

            def recurse(tree, where, **kwargs):
                __all_fields__ = {
                    **tree.__dataclass_fields__,
                    **tree.__dict__.get("__treeclass_fields__", {}),
                }
                for i, fld in enumerate(__all_fields__.values()):

                    cur_node = tree.__dict__[fld.name]
                    if not is_excluded(fld, cur_node) and is_treeclass(cur_node):
                        if fld.name == where:
                            # broadcast True to all subtrees
                            tree.__dict__[fld.name] = jtu.tree_map(set_true, cur_node)
                        else:
                            recurse(cur_node, where, **kwargs)
                    else:
                        tree.__dict__[fld.name] = (
                            set_true(cur_node)
                            if ((fld.name == where))
                            else set_false(cur_node)
                        )
                return tree

            return recurse(tree_copy, where, **kwargs)

        @inner_wrapper.register(type)
        def _(tree, where, **kwargs):
            """Filter by type"""
            tree_copy = jtu.tree_unflatten(*jtu.tree_flatten(tree)[::-1])

            def recurse(tree, where, **kwargs):

                __all_fields__ = {
                    **tree.__dataclass_fields__,
                    **tree.__dict__.get("__treeclass_fields__", {}),
                }
                for i, fld in enumerate(__all_fields__.values()):

                    cur_node = tree.__dict__[fld.name]

                    if not is_excluded(fld, cur_node) and is_treeclass(cur_node):
                        if isinstance(cur_node, where):
                            tree.__dict__[fld.name] = jtu.tree_map(set_true, cur_node)
                        else:
                            recurse(cur_node, where, **kwargs)
                    else:
                        tree.__dict__[fld.name] = (
                            set_true(cur_node)
                            if (isinstance(cur_node, where))
                            else set_false(cur_node)
                        )

                return tree

            return recurse(tree_copy, where, **kwargs)

        @inner_wrapper.register(dict)
        def _(tree, where, **kwargs):
            """Filter by metadata"""
            tree_copy = jtu.tree_unflatten(*jtu.tree_flatten(tree)[::-1])

            def in_metadata(fld):
                kws, vals = zip(*where.items())
                return all(
                    fld.metadata.get(kw, False) and (fld.metadata[kw] == val)
                    for kw, val in zip(kws, vals)
                )

            def recurse(tree, where, **kwargs):

                __all_fields__ = {
                    **tree.__dataclass_fields__,
                    **tree.__dict__.get("__treeclass_fields__", {}),
                }
                for i, fld in enumerate(__all_fields__.values()):

                    cur_node = tree.__dict__[fld.name]

                    if not is_excluded(fld, cur_node) and is_treeclass(cur_node):
                        if in_metadata(fld):
                            tree.__dict__[fld.name] = jtu.tree_map(set_true, cur_node)
                        else:
                            recurse(cur_node, where, **kwargs)
                    else:
                        tree.__dict__[fld.name] = (
                            set_true(cur_node)
                            if in_metadata(fld)
                            else set_false(cur_node)
                        )

                return tree

            return recurse(tree_copy, where, **kwargs)

        return (
            jtu.tree_map(node_not, inner_wrapper(self, rhs))
            if func == op.ne
            else inner_wrapper(self, rhs)
        )

    return wrapper


def _append_exception(func):
    def exception(*args, **kwargs):
        raise Exception(f"{func} is not supported.")

    return exception


class treeOpBase:

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
