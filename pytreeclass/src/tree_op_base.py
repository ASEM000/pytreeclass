from __future__ import annotations

import functools
import operator as op

import jax.numpy as jnp
import jax.tree_util as jtu


def _append_math_op(func):
    """binary and unary magic operations"""

    @functools.wraps(func)
    def call(self, rhs=None):

        if rhs is None:  # unary operation
            return jtu.tree_map(lambda x: func(x), self)

        elif isinstance(rhs, (int, float, complex, bool)):  # binary operation
            return (
                jtu.tree_map(lambda x: func(x, rhs), self) if rhs is not None else self
            )

        elif isinstance(rhs, type(self)):  # class instance
            return jtu.tree_map(
                lambda x, y: func(x, y) if y is not None else x, self, rhs
            )

        else:
            raise NotImplementedError(f"Found type(rhs) = {type(rhs)}")

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

    def __or__(self, rhs):
        def node_or(x, y):
            return x if isinstance(x, jnp.ndarray) else (x or y)

        return jtu.tree_map(node_or, self, rhs, is_leaf=lambda x: x is None)
