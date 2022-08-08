from __future__ import annotations

import copy
import functools
import operator as op

import jax.numpy as jnp
import jax.tree_util as jtu

import pytreeclass.src.tree_util as ptu
from pytreeclass.src.decorator_util import dispatch


def _append_math_op(func):
    """binary and unary magic operations"""

    @functools.wraps(func)
    def wrapper(self, rhs=None):
        @dispatch(argnum=1)
        def inner_wrapper(self, rhs):
            raise NotImplementedError((f"rhs of type {type(rhs)} is not implemented."))

        @inner_wrapper.register(type(None))
        def _(self, rhs):
            return jtu.tree_map(lambda x: func(x), self)

        @inner_wrapper.register(int)
        @inner_wrapper.register(float)
        @inner_wrapper.register(complex)
        @inner_wrapper.register(bool)
        def _(self, rhs):
            return (
                jtu.tree_map(lambda x: func(x, rhs), self) if rhs is not None else self
            )

        @inner_wrapper.register(type(self))
        def _(self, rhs):
            return jtu.tree_map(
                lambda x, y: func(x, y) if y is not None else x, self, rhs
            )

        return inner_wrapper(self, rhs)

    return wrapper


def _append_math_eq_ne(func):
    """Append eq/ne operations"""
    assert func in [op.eq, op.ne], f"func={func} is not implemented."

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

    @functools.wraps(func)
    def wrapper(self, rhs):
        @dispatch(argnum=1)
        def inner_wrapper(tree, where, **kwargs):
            raise NotImplementedError(
                f"where of type {type(where)} is not implemented."
            )

        @inner_wrapper.register(int)
        @inner_wrapper.register(float)
        @inner_wrapper.register(complex)
        @inner_wrapper.register(bool)
        def _(self, rhs):
            return (
                jtu.tree_map(lambda x: func(x, rhs), self) if rhs is not None else self
            )

        @inner_wrapper.register(type(self))
        def _(self, rhs):
            return jtu.tree_map(
                lambda x, y: func(x, y) if y is not None else x, self, rhs
            )

        @inner_wrapper.register(str)
        def _(tree, where, **kwargs):
            """Filter by field name"""
            tree_copy = copy.deepcopy(tree)

            def recurse(tree, where, **kwargs):
                for i, fld in enumerate(tree.__dataclass_fields__.values()):

                    cur_node = tree.__dict__[fld.name]
                    if not ptu.is_excluded(fld, tree) and ptu.is_treeclass(cur_node):
                        if func(fld.name, where):
                            tree.__dict__[fld.name] = jtu.tree_map(set_true, cur_node)
                        else:
                            recurse(cur_node, where, **kwargs)
                    else:
                        tree.__dict__[fld.name] = (
                            set_true(cur_node)
                            if (func(fld.name, where))
                            else set_false(cur_node)
                        )
                return tree

            return recurse(tree_copy, where, **kwargs)

        @inner_wrapper.register(type)
        def _(tree, where, **kwargs):
            """Filter by type"""
            tree_copy = copy.deepcopy(tree)

            def instance_func(lhs, rhs):
                if func == op.eq:
                    return isinstance(lhs, rhs)
                elif func == op.ne:
                    return not isinstance(lhs, rhs)

            def recurse(tree, where, **kwargs):
                for i, fld in enumerate(tree.__dataclass_fields__.values()):
                    cur_node = tree.__dict__[fld.name]

                    if not ptu.is_excluded(fld, tree) and ptu.is_treeclass(cur_node):
                        if instance_func(cur_node, where):
                            tree.__dict__[fld.name] = jtu.tree_map(set_true, cur_node)
                        else:
                            recurse(cur_node, where, **kwargs)
                    else:
                        tree.__dict__[fld.name] = (
                            set_true(cur_node)
                            if (instance_func(cur_node, where))
                            else set_false(cur_node)
                        )

                return tree

            return recurse(tree_copy, where, **kwargs)

        @inner_wrapper.register(dict)
        def _(tree, where, **kwargs):
            """Filter by metadata"""
            tree_copy = copy.deepcopy(tree)
            kws, vals = zip(*where.items())

            in_meta = lambda fld: all(kw in fld.metadata for kw in kws) and all(
                func(fld.metadata[kw], val) for kw, val in zip(kws, vals)
            )

            def recurse(tree, where, **kwargs):
                for i, fld in enumerate(tree.__dataclass_fields__.values()):
                    cur_node = tree.__dict__[fld.name]

                    if not ptu.is_excluded(fld, tree) and ptu.is_treeclass(cur_node):
                        if in_meta(fld):
                            tree.__dict__[fld.name] = jtu.tree_map(set_true, cur_node)
                        else:
                            recurse(cur_node, where, **kwargs)
                    else:
                        tree.__dict__[fld.name] = (
                            set_true(cur_node) if in_meta(fld) else set_false(cur_node)
                        )

                return tree

            return recurse(tree_copy, where, **kwargs)

        return inner_wrapper(self, rhs)

    return wrapper


class treeOpBase:

    __abs__ = _append_math_op(op.abs)
    __add__ = _append_math_op(op.add)
    __radd__ = _append_math_op(op.add)
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
    __pos__ = _append_math_op(op.pos)
    __pow__ = _append_math_op(op.pow)
    __rshift__ = _append_math_op(op.rshift)
    __sub__ = _append_math_op(op.sub)
    __rsub__ = _append_math_op(op.sub)
    __truediv__ = _append_math_op(op.truediv)
    __xor__ = _append_math_op(op.xor)

    def __or__(self, rhs):
        def node_or(x, y):
            if isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):
                if x.shape == y.shape:
                    return jnp.logical_or(x, y)
                elif jnp.array_equal(x, jnp.array([])):
                    return y
                elif jnp.array_equal(y, jnp.array([])):
                    return x
                else:
                    raise ValueError("Cannot or arrays of different shapes")
            else:
                return x or y

        return jtu.tree_map(node_or, self, rhs, is_leaf=lambda x: x is None)
