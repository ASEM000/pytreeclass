from __future__ import annotations

import copy
import dataclasses as dc
import math
import operator as op
from typing import Any

import jax.tree_util as jtu
import numpy as np

from pytreeclass._src.tree_operator import _hash_node

PyTree = Any


class _Wrapper:
    def __init__(self, x: Any):
        # disable composition of Wrappers
        self.__wrapped__ = x.unwrap() if isinstance(x, _Wrapper) else x

    def unwrap(self):
        return self.__wrapped__


class _HashableWrapper(_Wrapper):
    # used to wrap metadata to make it hashable
    # this is intended to wrap frozen values to avoid error when comparing
    # the metadata.
    def __eq__(self, rhs: Any) -> bool:
        if not isinstance(rhs, _HashableWrapper):
            return False
        return _hash_node(self.unwrap()) == _hash_node(rhs.unwrap())

    def __hash__(self):
        return _hash_node(self.unwrap())


@jtu.register_pytree_node_class
class FrozenWrapper(_Wrapper):
    # a transparent wrapper that freezes the wrapped object setter and inplace operations
    # plus returning `None` for `jtu.tree_leaves`.
    # however, when interacting with it, the returned values is not frozen.
    __call__ = lambda self, *a, **k: self.unwrap()(*a, **k)

    # delegate most methods to the wrapped object
    # however, the returned values are not frozen
    # and inplace operations are not allowed
    # by performing the delegated operation on a copy of the wrapped object
    __getattr__ = lambda self, k: getattr(copy.copy(self.unwrap()), k)

    # class methods that can not be delegated with `__getattr__`
    # and need to be explicitly defined

    __abs__ = lambda self: op.abs(self.unwrap())
    __add__ = lambda self, rhs: op.add(self.unwrap(), rhs)
    __and__ = lambda self, rhs: op.and_(self.unwrap(), rhs)
    __bool__ = lambda self: bool(self.unwrap())
    __ceil__ = lambda self: math.ceil(self.unwrap())
    __divmod__ = lambda self, rhs: divmod(self.unwrap(), rhs)
    __eq__ = lambda self, rhs: op.eq(self.unwrap(), rhs)
    __float__ = lambda self: float(self.unwrap())
    __floor__ = lambda self: math.floor(self.unwrap())
    __floordiv__ = lambda self, rhs: op.floordiv(self.unwrap(), rhs)
    __ge__ = lambda self, rhs: op.ge(self.unwrap(), rhs)
    __gt__ = lambda self, rhs: op.gt(self.unwrap(), rhs)
    __int__ = lambda self: int(self.unwrap())
    __inv__ = lambda self: op.inv(self.unwrap())
    __invert__ = lambda self: op.invert(self.unwrap())
    __le__ = lambda self, rhs: op.le(self.unwrap(), rhs)
    __lshift__ = lambda self, rhs: op.lshift(self.unwrap(), rhs)
    __lt__ = lambda self, rhs: op.lt(self.unwrap(), rhs)
    __matmul__ = lambda self, rhs: op.matmul(self.unwrap(), rhs)
    __mod__ = lambda self, rhs: op.mod(self.unwrap(), rhs)
    __mul__ = lambda self, rhs: op.mul(self.unwrap(), rhs)
    __ne__ = lambda self, rhs: op.ne(self.unwrap(), rhs)
    __neg__ = lambda self, rhs: op.neg(self.unwrap())
    __or__ = lambda self, rhs: op.or_(self.unwrap(), rhs)
    __pos__ = lambda self: op.pos(self.unwrap())
    __pow__ = lambda self, rhs: op.pow(self.unwrap(), rhs)
    __radd__ = lambda self, rhs: op.add(rhs, self.unwrap())
    __rand__ = lambda self, rhs: op.and_(rhs, self.unwrap())
    __rdivmod__ = lambda self, rhs: divmod(rhs, self.unwrap())
    __rfloordiv__ = lambda self, rhs: op.floordiv(rhs, self.unwrap())
    __rlshift__ = lambda self, rhs: op.lshift(rhs, self.unwrap())
    __rmod__ = lambda self, rhs: op.mod(rhs, self.unwrap())
    __rmul__ = lambda self, rhs: op.mul(rhs, self.unwrap())
    __rmatmul__ = lambda self, rhs: op.matmul(rhs, self.unwrap())
    __ror__ = lambda self, rhs: op.or_(rhs, self.unwrap())
    __round__ = lambda self, rhs: round(self.unwrap(), rhs)
    __rpow__ = lambda self, rhs: op.pow(rhs, self.unwrap())
    __rrshift__ = lambda self, rhs: op.rshift(rhs, self.unwrap())
    __rshift__ = lambda self, rhs: op.rshift(self.unwrap(), rhs)
    __rsub__ = lambda self, rhs: op.sub(rhs, self.unwrap())
    __rtruediv__ = lambda self, rhs: op.truediv(rhs, self.unwrap())
    __rxor__ = lambda self, rhs: op.xor(rhs, self.unwrap())
    __sub__ = lambda self, rhs: op.sub(self.unwrap(), rhs)
    __truediv__ = lambda self, rhs: op.truediv(self.unwrap(), rhs)
    __trunk__ = lambda self: math.trunk(self.unwrap())
    __xor__ = lambda self, rhs: op.xor(self.unwrap(), rhs)
    __hash__ = lambda self: hash(self.unwrap())

    # repr and str
    __repr__ = lambda self: f"#{self.unwrap()!r}"
    __str__ = lambda self: f"#{self.unwrap()!s}"

    # JAX methods
    def tree_flatten(self):
        # Wrapping the metadata to ensure its hashability and equality
        # https://github.com/google/jax/issues/13027
        return (None,), _HashableWrapper(self.unwrap())

    @classmethod
    def tree_unflatten(cls, treedef, leaves):
        self = object.__new__(cls)
        self.__dict__.update(__wrapped__=treedef.unwrap())
        return self

    def __setattr__(self, key, value):
        # allow setting the wrapped value only once.
        if "__wrapped__" in self.__dict__:
            raise dc.FrozenInstanceError("Cannot assign to frozen instance.")
        else:
            super().__setattr__(key, value)


def freeze(x: Any) -> FrozenWrapper:
    """Wrap a value in a FrozenWrapper

    Example:
        >>> frozen_value = freeze(1)
        >>> frozen_value
        #1

        # When interacting with the wrapped value, the returned value is **not frozen**
        >>> frozen_value + 1
        2

        >>> # wrapped value is considred for mean calculation,
        >>> # but gradient is ignored
        @jax.grad
        >>> def f(x):
        ...    return jnp.mean(jnp.asarray(x)**2)

        >>> f([2.,2.,pytc.freeze(2.)])
        [Array(1.3333334, dtype=float32, weak_type=True), Array(1.3333334, dtype=float32, weak_type=True), #2.0]


        # Inplace operations take no effect
        >>> frozen_value = freeze([1, 2, 3])
        >>> frozen_value.append(4)
        >>> frozen_value
        #[1, 2, 3]


        # using `frozen` with `jax.jit` and `jax.grad`
        >>> @ft.partial(jax.grad,argnums=0)
        ... def dfdx(x,y):
        ...    # we are taking derivative w.r.t x, however x is a frozen value
        ...    # so the gradient of x is **ignored** and x is returned as is
        ...    # without using `pytc.freeze` we would have to define `static_argnums=0` in `jit`
        ...    return x+y**2

        >>> dfdx(pytc.freeze(1),2.)
        #1

        >>> @ft.partial(jax.grad,argnums=1)
        ... def dfdy(x,y):
        ...    # we are taking derivative w.r.t y,
        ...    # so the gradient of y is just 2*y
        ...    return x+y**2

        >>> dfdy(pytc.freeze(1),2.)
        Array(4., dtype=float32, weak_type=True)
    """
    return FrozenWrapper(x)


def unfreeze(x: Any) -> Any:
    """Unfreeze `frozen` value.

    **use `is_leaf=pytc.is_frozen` with `jax.tree_util.tree_map` to unfreeze a tree.**

    Example:
        >>> frozen_value = pytc.freeze(1)
        >>> pytc.unfreeze(frozen_value)
        1

        # usage with `jax.tree_map`
        >>> frozen_tree = jtu.tree_map(pytc.freeze, {"a": 1, "b": 2})
        >>> unfrozen_tree = jtu.tree_map(pytc.unfreeze, frozen_tree, is_leaf=pytc.is_frozen)
        >>> unfrozen_tree
        {'a': 1, 'b': 2}
    """
    return x.unwrap() if isinstance(x, FrozenWrapper) else x


def is_frozen(node: Any) -> bool:
    """Check if a node is frozen"""
    return isinstance(node, FrozenWrapper)


def is_nondiff(node: Any) -> bool:
    """Returns False if the node is a float, complex number, or a numpy array of floats or complex numbers."""
    # this is meant to be used with `jtu.tree_map`.

    if hasattr(node, "dtype") and np.issubdtype(node.dtype, np.inexact):
        return False
    if isinstance(node, (float, complex)):
        return False
    return True
