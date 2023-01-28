from __future__ import annotations

import functools as ft
import operator as op

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.core import Tracer

"""A wrapper around a tree that allows to use the tree leaves as if they were scalars."""


def _hash_node(node):
    if hasattr(node, "dtype") and hasattr(node, "shape"):
        return hash(np.array(node).tobytes())
    if isinstance(node, set):
        return hash(frozenset(node))
    if isinstance(node, dict):
        return hash(frozenset(node.items()))
    if isinstance(node, list):
        return hash(tuple(node))
    return hash(node)


def _hash(tree):
    hashed = jtu.tree_map(_hash_node, jtu.tree_leaves(tree))
    return hash((*hashed, jtu.tree_structure(tree)))


def _append_math_op(func):
    """Binary and unary magic operations"""

    @ft.wraps(func)
    def wrapper(lhs, rhs=None, is_leaf=None):
        """`jtu.tree_map` for unary/binary operators broadcasting"""
        if isinstance(rhs, type(lhs)):
            # rhs is a tree of the same type as lhs then we use the tree_map to apply the operator leaf-wise
            return jtu.tree_map(func, lhs, rhs, is_leaf=is_leaf)

        if isinstance(rhs, (Tracer, jnp.ndarray, int, float, complex, bool, str)):
            # if rhs is a scalar then we use the tree_map to apply the operator with broadcasting the rhs
            return jtu.tree_map(lambda x: func(x, rhs), lhs, is_leaf=is_leaf)

        if isinstance(rhs, type(None)):
            # Unary operator case
            return jtu.tree_map(func, lhs, is_leaf=is_leaf)
        raise NotImplementedError(f"rhs of type {type(rhs)} is not implemented.")

    return wrapper


class _TreeOperator:
    """Base class for tree operators used

    Example:
        >>> import jax.tree_util as jtu
        >>> import dataclasses as dc
        >>> @jtu.register_pytree_node_class
        ... @dc.dataclass
        ... class Tree(_TreeOperator):
        ...    a: int =1
        ...    def tree_flatten(self):
        ...        return (self.a,), None
        ...    @classmethod
        ...    def tree_unflatten(cls, _, children):
        ...        return cls(*children)

        >>> tree = Tree()
        >>> tree + 1
        Tree(a=2)
    """

    __hash__ = _hash  # hash the tree
    __abs__ = _append_math_op(op.abs)  # abs the tree leaves
    __add__ = _append_math_op(op.add)  # add to the tree leaves
    __radd__ = _append_math_op(op.add)  # add to the tree leaves
    __and__ = _append_math_op(op.and_)  # and the tree leaves
    __rand__ = _append_math_op(op.and_)  # and the tree leaves
    __eq__ = _append_math_op(op.eq)  # = the tree leaves
    __floordiv__ = _append_math_op(op.floordiv)  # // the tree leaves
    __ge__ = _append_math_op(op.ge)  # >= the tree leaves
    __gt__ = _append_math_op(op.gt)  # > the tree leaves
    __inv__ = _append_math_op(op.inv)  # ~ the tree leaves
    __invert__ = _append_math_op(op.invert)  # invert the tree leaves
    __le__ = _append_math_op(op.le)  # <= the tree leaves
    __lshift__ = _append_math_op(op.lshift)  # lshift the tree leaves
    __lt__ = _append_math_op(op.lt)  # < the tree leaves
    __matmul__ = _append_math_op(op.matmul)  # matmul the tree leaves
    __mod__ = _append_math_op(op.mod)  # % the tree leaves
    __mul__ = _append_math_op(op.mul)  # * the tree leaves
    __rmul__ = _append_math_op(op.mul)  # * the tree leaves
    __ne__ = _append_math_op(op.ne)  # != the tree leaves
    __neg__ = _append_math_op(op.neg)  # - the tree leaves
    __not__ = _append_math_op(op.not_)  # not the tree leaves
    __or__ = _append_math_op(op.or_)  # or the tree leaves
    __pos__ = _append_math_op(op.pos)  # + the tree leaves
    __pow__ = _append_math_op(op.pow)  # ** the tree leaves
    __rshift__ = _append_math_op(op.rshift)  # rshift the tree leaves
    __sub__ = _append_math_op(op.sub)  # - the tree leaves
    __rsub__ = _append_math_op(op.sub)  # - the tree leaves
    __truediv__ = _append_math_op(op.truediv)  # / the tree leaves
