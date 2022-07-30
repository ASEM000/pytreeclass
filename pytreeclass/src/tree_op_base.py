from __future__ import annotations

import functools
import operator as op

import jax.numpy as jnp
from jax.tree_util import tree_map, tree_reduce


def _append_math_op(func):
    """binary and unary magic operations"""

    @functools.wraps(func)
    def call(self, rhs=None):

        if rhs is None:  # unary operation
            return tree_map(lambda x: func(x), self)

        elif isinstance(rhs, (int, float, complex, bool)):  # binary operation
            return tree_map(lambda x: func(x, rhs), self) if rhs is not None else self

        elif isinstance(rhs, type(self)):  # class instance
            return tree_map(lambda x, y: func(x, y) if y is not None else x, self, rhs)

        else:
            raise NotImplementedError(f"Found type(rhs) = {type(rhs)}")

    return call


def _append_numpy_op(func):
    """array operations"""

    @functools.wraps(func)
    def call(self, *args, **kwargs):
        return tree_map(lambda node: func(node, *args, **kwargs), self)

    return call


def _append_reduced_numpy_op(func, reduce_op, init_val):
    """reduced array operations"""

    @functools.wraps(func)
    def call(self, *args, **kwargs):
        return tree_reduce(
            lambda acc, cur: reduce_op(acc, func(cur, *args, **kwargs)), self, init_val
        )

    return call


class treeOpBase:

    __abs__ = _append_math_op(op.abs)
    __add__ = _append_math_op(op.add)
    __radd__ = _append_math_op(op.add)
    __eq__ = _append_math_op(op.eq)
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
    __ne__ = _append_math_op(op.ne)
    __neg__ = _append_math_op(op.neg)
    __not__ = _append_math_op(op.not_)
    __pos__ = _append_math_op(op.pos)
    __pow__ = _append_math_op(op.pow)
    __rshift__ = _append_math_op(op.rshift)
    __sub__ = _append_math_op(op.sub)
    __rsub__ = _append_math_op(op.sub)
    __truediv__ = _append_math_op(op.truediv)
    __xor__ = _append_math_op(op.xor)

    imag = property(_append_numpy_op(jnp.imag))
    real = property(_append_numpy_op(jnp.real))
    conj = property(_append_numpy_op(jnp.conj))

    abs = _append_numpy_op(jnp.abs)
    amax = _append_numpy_op(jnp.amax)
    amin = _append_numpy_op(jnp.amin)
    arccos = _append_numpy_op(jnp.arccos)
    arcsin = _append_numpy_op(jnp.arcsin)
    sum = _append_numpy_op(jnp.sum)
    prod = _append_numpy_op(jnp.prod)
    mean = _append_numpy_op(jnp.mean)

    reduce_abs = _append_reduced_numpy_op(jnp.abs, op.add, 0)
    reduce_amax = _append_reduced_numpy_op(jnp.amax, op.add, 0)
    reduce_amin = _append_reduced_numpy_op(jnp.amin, op.add, 0)
    reduce_arccos = _append_reduced_numpy_op(jnp.arccos, op.add, 0)
    reduce_arcsin = _append_reduced_numpy_op(jnp.arcsin, op.add, 0)
    reduce_sum = _append_reduced_numpy_op(jnp.sum, op.add, 0)
    reduce_prod = _append_reduced_numpy_op(jnp.prod, op.mul, 1)
    reduce_mean = _append_reduced_numpy_op(jnp.mean, op.add, 0)

    def __or__(self, rhs):
        def node_or(x, y):
            return x if isinstance(x, jnp.ndarray) else (x or y)

        return tree_map(node_or, self, rhs, is_leaf=lambda x: x is None)

    def register_op(self, func, *, name, reduce_op=None, init_val=None):
        """register a math operation"""

        def element_call(*args, **kwargs):
            return tree_map(lambda node: func(node, *args, **kwargs), self)

        setattr(self, name, element_call)

        if (reduce_op is not None) and (init_val is not None):

            def reduced_call(*args, **kwargs):
                return tree_reduce(
                    lambda acc, cur: reduce_op(acc, func(cur, *args, **kwargs)),
                    self,
                    init_val,
                )

            setattr(self, f"reduce_{name}", reduced_call)
