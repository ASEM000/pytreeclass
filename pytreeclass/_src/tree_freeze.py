from __future__ import annotations

import hashlib
from typing import Any

import jax
import jax.tree_util as jtu
import numpy as np

PyTree = Any


def _hash_node(node: Any) -> int:
    if isinstance(node, (jax.Array, np.ndarray)):
        return int(hashlib.sha256(np.array(node).tobytes()).hexdigest(), 16)
    if isinstance(node, set):
        return hash(frozenset(node))
    if isinstance(node, dict):
        return hash(frozenset(node.items()))
    if isinstance(node, list):
        return hash(tuple(node))
    return hash(node)


def tree_hash(*trees: PyTree) -> int:
    hashed = jtu.tree_map(_hash_node, jtu.tree_leaves(trees))
    return hash((*hashed, jtu.tree_structure(trees)))


class _ImmutableWrapper:
    __slots__ = ("__wrapped__", "__weakref__")

    def __init__(self, x: Any) -> None:
        object.__setattr__(self, "__wrapped__", getattr(x, "__wrapped__", x))

    def unwrap(self) -> Any:
        return getattr(self, "__wrapped__")

    def __setattr__(self, _, __) -> None:
        raise AttributeError("Cannot assign to frozen instance.")

    def __delattr__(self, _: str) -> None:
        raise AttributeError("Cannot delete from frozen instance.")


class _HashableWrapper(_ImmutableWrapper):
    def __eq__(self, rhs: Any) -> bool:
        if not isinstance(rhs, _HashableWrapper):
            return False
        return _hash_node(self.unwrap()) == _hash_node(rhs.unwrap())

    def __hash__(self) -> int:
        return tree_hash(self.unwrap())


def _frozen_error(opname: str, tree):
    raise NotImplementedError(
        f"Cannot apply `{opname}` operation a frozen object `{tree!r}`.\n"
        "Unfreeze the object first to apply operations to it\n"
        "Example:\n"
        ">>> import jax\n"
        ">>> import pytreeclass as pytc\n"
        ">>> tree = jax.tree_map(pytc.unfreeze, tree, is_leaf=pytc.is_frozen)"
    )


class _FrozenWrapper(_ImmutableWrapper):
    def __repr__(self):
        return f"#{self.unwrap()!r}"

    def __eq__(self, rhs: Any) -> bool:
        if not isinstance(rhs, _FrozenWrapper):
            return False
        return self.unwrap() == rhs.unwrap()

    def __hash__(self) -> int:
        return tree_hash(self.unwrap())

    # raise helpful error message when trying to interact with frozen object
    __add__ = __radd__ = __iadd__ = lambda x, _: _frozen_error("+", x)
    __sub__ = __rsub__ = __isub__ = lambda x, _: _frozen_error("-", x)
    __mul__ = __rmul__ = __imul__ = lambda x, _: _frozen_error("*", x)
    __matmul__ = __rmatmul__ = __imatmul__ = lambda x, _: _frozen_error("@", x)
    __truediv__ = __rtruediv__ = __itruediv__ = lambda x, _: _frozen_error("/", x)
    __floordiv__ = __rfloordiv__ = __ifloordiv__ = lambda x, _: _frozen_error("//", x)
    __mod__ = __rmod__ = __imod__ = lambda x, _: _frozen_error("%", x)
    __pow__ = __rpow__ = __ipow__ = lambda x, _: _frozen_error("**", x)
    __lshift__ = __rlshift__ = __ilshift__ = lambda x, _: _frozen_error("<<", x)
    __rshift__ = __rrshift__ = __irshift__ = lambda x, _: _frozen_error(">>", x)
    __and__ = __rand__ = __iand__ = lambda x, _: _frozen_error("and", x)
    __xor__ = __rxor__ = __ixor__ = lambda x, _: _frozen_error("xor", x)
    __or__ = __ror__ = __ior__ = lambda x, _: _frozen_error("or", x)
    __neg__ = __pos__ = __abs__ = __invert__ = lambda x: _frozen_error("unary op", x)
    __call__ = lambda x, *_, **__: _frozen_error("call", x)


jtu.register_pytree_node(
    nodetype=_FrozenWrapper,
    flatten_func=lambda tree: ((), _HashableWrapper(tree.unwrap())),
    unflatten_func=lambda treedef, _: _FrozenWrapper(treedef.unwrap()),
)


def freeze(wrapped: Any) -> _FrozenWrapper:
    r"""Freeze a value to avoid updating it by `jax` transformations.

    Example:
        >>> import jax
        >>> import pytreeclass as pytc
        >>> import jax.tree_util as jtu
        >>> # Usage with `jax.tree_util.tree_leaves`
        >>> # no leaves for a wrapped value
        >>> jtu.tree_leaves(pytc.freeze(2.))
        []

        >>> # retrieve the frozen wrapper value using `is_leaf=pytc.is_frozen`
        >>> jtu.tree_leaves(pytc.freeze(2.), is_leaf=pytc.is_frozen)
        [#2.0]

        >>> # Usage with `jax.tree_util.tree_map`
        >>> a= [1,2,3]
        >>> a[1] = pytc.freeze(a[1])
        >>> jtu.tree_map(lambda x:x+100, a)
        [101, #2, 103]

        >>> class Test(pytc.TreeClass):
        ...     a: float
        ...     @jax.value_and_grad
        ...     def __call__(self, x):
        ...         # unfreeze `a` to update it
        ...         self = jax.tree_util.tree_map(pytc.unfreeze, self, is_leaf=pytc.is_frozen)
        ...         return x ** self.a

        >>> # without `freeze` wrapping `a`, `a` will be updated
        >>> value, grad = Test(a = 2.)(2.)
        >>> print(f"value: {value}\ngrad: {grad}")
        value: 4.0
        grad: Test(a=2.7725887)

        >>> # with `freeze` wrapping `a`, `a` will NOT be updated
        >>> value, grad = Test(a=pytc.freeze(2.))(2.)
        >>> print(f"value: {value}\ngrad: {grad}")
        value: 4.0
        grad: Test(a=#2.0)

        >>> # usage with `jax.tree_map` to freeze a tree
        >>> tree = Test(a = 2.)
        >>> frozen_tree = jax.tree_map(pytc.freeze, tree)
        >>> value, grad = frozen_tree(2.)
        >>> print(f"value: {value}\ngrad: {grad}")
        value: 4.0
        grad: Test(a=#2.0)
    """
    return _FrozenWrapper(wrapped)


def unfreeze(x: Any) -> Any:
    """Unfreeze `frozen` value, otherwise return the value itself.

    - use `is_leaf=pytc.is_frozen` with `jax.tree_util.tree_map` to unfreeze a tree.**

    Example:
        >>> import pytreeclass as pytc
        >>> import jax.tree_util as jtu
        >>> frozen_value = pytc.freeze(1)
        >>> pytc.unfreeze(frozen_value)
        1
        >>> # usage with `jax.tree_map`
        >>> frozen_tree = jtu.tree_map(pytc.freeze, {"a": 1, "b": 2})
        >>> unfrozen_tree = jtu.tree_map(pytc.unfreeze, frozen_tree, is_leaf=pytc.is_frozen)
        >>> unfrozen_tree
        {'a': 1, 'b': 2}
    """
    return x.unwrap() if isinstance(x, _FrozenWrapper) else x


def is_frozen(wrapped: Any) -> bool:
    """Returns True if the value is a frozen wrapper."""
    return isinstance(wrapped, _FrozenWrapper)


def is_nondiff(x: Any) -> bool:
    """Returns False if the node is a float, complex number, or a numpy array of floats or complex numbers.

    Example:
        >>> import pytreeclass as pytc
        >>> import jax.numpy as jnp
        >>> pytc.is_nondiff(jnp.array(1))  # int array is non-diff type
        True
        >>> pytc.is_nondiff(jnp.array(1.))  # float array is diff type
        False
        >>> pytc.is_nondiff(1)  # int is non-diff type
        True
        >>> pytc.is_nondiff(1.)  # float is diff type
        False

    Note:
        This function is meant to be used with `jax.tree_util.tree_map` to create a mask
        for non-differentiable nodes in a tree, that can be used to freeze the non-differentiable nodes
        in a tree.
    """
    if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.inexact):
        return False
    if isinstance(x, (float, complex)):
        return False
    return True
