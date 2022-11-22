from __future__ import annotations

import dataclasses
import operator as op
from typing import Any, Callable, Iterable

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

import pytreeclass._src.dataclass_util as dcu
from pytreeclass._src.dataclass_util import _dataclass_structure, _fieldDict, _mutable
from pytreeclass._src.tree_indexer import _at_indexer
from pytreeclass._src.tree_op import _append_math_eq_ne, _append_math_op, _eq, _ne
from pytreeclass.tree_viz.tree_pprint import tree_repr, tree_str

PyTree = Any


def _setattr(tree, key: str, value: Any) -> None:
    if tree.__dataclass_params__.frozen:
        msg = f"Cannot set {key}={value!r}. Use `.at['{key}'].set({value!r})` instead."
        raise dataclasses.FrozenInstanceError(msg)

    object.__setattr__(tree, key, value)

    if dataclasses.is_dataclass(value) and (
        key not in [f.name for f in dataclasses.fields(tree)]
    ):
        field_item = dataclasses.field()
        object.__setattr__(field_item, "name", key)
        object.__setattr__(field_item, "type", type(value))
        object.__setattr__(field_item, "_field_type", dataclasses._FIELD)

        # register it to dataclass fields
        tree.__dataclass_fields__[key] = field_item


def _delattr(tree, key: str) -> None:
    if tree.__dataclass_params__.frozen:
        raise dataclasses.FrozenInstanceError(f"Cannot delete {key}.")
    object.__delattr__(tree, key)


def _new(cls, *a, **k):
    tree = object.__new__(cls)

    _params = dataclasses._DataclassParams(
        init=tree.__dataclass_params__.init,
        repr=tree.__dataclass_params__.repr,
        eq=tree.__dataclass_params__.eq,
        order=tree.__dataclass_params__.order,
        unsafe_hash=tree.__dataclass_params__.unsafe_hash,
        frozen=tree.__dataclass_params__.frozen,
    )

    _dataclass_fields = {
        field_item.name: dcu.field_copy(field_item)
        for field_item in dataclasses.fields(tree)
    }

    object.__setattr__(tree, "__dataclass_params__", _params)
    object.__setattr__(tree, "__dataclass_fields__", _dataclass_fields)

    for field_item in dataclasses.fields(tree):
        if field_item.default is not dataclasses.MISSING:
            object.__setattr__(tree, field_item.name, field_item.default)
    return tree


def _hash(tree):
    """Return a hash of the tree"""

    def _hash_node(node):
        """hash the leaves of the tree"""
        if isinstance(node, jnp.ndarray):
            return np.array(node).tobytes()
        elif isinstance(node, set):
            # jtu.tree_map does not traverse sets
            return frozenset(node)
        return node

    return hash(
        (*jtu.tree_map(_hash_node, jtu.tree_leaves(tree)), jtu.tree_structure(tree))
    )


def _copy(tree):
    return jtu.tree_unflatten(*jtu.tree_flatten(tree)[::-1])


def _flatten(tree) -> tuple[Any, tuple[str, _fieldDict[str, Any]]]:
    """Flatten rule for `jax.tree_flatten`"""
    dynamic, static = _dataclass_structure(tree)
    return dynamic.values(), (dynamic.keys(), static)


def _unflatten(cls, treedef, leaves):
    """Unflatten rule for `jax.tree_unflatten`"""
    tree = object.__new__(cls)
    # update the instance values with the retrieved dynamic and static values
    tree.__dict__.update(dict(zip(treedef[0], leaves)))
    tree.__dict__.update(treedef[1])
    return tree


def treeclass(cls):
    """Decorator to make a class a treeclass"""
    dcls = dataclasses.dataclass(
        init="__init__" not in vars(cls),  # if __init__ is defined, do not overwrite it
        repr=False,  # repr is handled by _treePretty
        eq=False,  # eq is handled by _treeOp
        order=False,  # order is handled by _treeOp
        unsafe_hash=False,  # unsafe_hash is handled by _treeOp
        frozen=True,  # frozen is handled by _setter/_delattr
    )(cls)

    attrs = dict(
        __new__=_new,  # overwrite __new__ to initialize instance variables
        __init__=_mutable(cls.__init__),  # make it mutable during initialization
        __setattr__=_setattr,  # disable direct attribute setting unless __immutable_treeclass__ is False
        __delattr__=_delattr,  # disable direct attribute deletion unless __immutable_treeclass__ is False
        __repr__=tree_repr,  # pretty print the tree representation
        __str__=tree_str,  # pretty print the tree
        __hash__=_hash,  # hash the tree
        __copy__=_copy,  # copy the tree
        __abs__=_append_math_op(op.abs),  # abs the tree leaves
        __add__=_append_math_op(op.add),  # add to the tree leaves
        __radd__=_append_math_op(op.add),  # add to the tree leaves
        __and__=_append_math_op(op.and_),  # and the tree leaves
        __rand__=_append_math_op(op.and_),  # and the tree leaves
        __eq__=_append_math_eq_ne(_eq),  # = the tree leaves
        __floordiv__=_append_math_op(op.floordiv),  # // the tree leaves
        __ge__=_append_math_op(op.ge),  # >= the tree leaves
        __gt__=_append_math_op(op.gt),  # > the tree leaves
        __inv__=_append_math_op(op.inv),  # ~ the tree leaves
        __invert__=_append_math_op(op.invert),  # invert the tree leaves
        __le__=_append_math_op(op.le),  # <= the tree leaves
        __lshift__=_append_math_op(op.lshift),  # lshift the tree leaves
        __lt__=_append_math_op(op.lt),  # < the tree leaves
        __matmul__=_append_math_op(op.matmul),  # matmul the tree leaves
        __mod__=_append_math_op(op.mod),  # % the tree leaves
        __mul__=_append_math_op(op.mul),  # * the tree leaves
        __rmul__=_append_math_op(op.mul),  # * the tree leaves
        __ne__=_append_math_eq_ne(_ne),  # != the tree leaves
        __neg__=_append_math_op(op.neg),  # - the tree leaves
        __not__=_append_math_op(op.not_),  # not the tree leaves
        __or__=_append_math_op(op.or_),  # or the tree leaves
        __pos__=_append_math_op(op.pos),  # + the tree leaves
        __pow__=_append_math_op(op.pow),  # ** the tree leaves
        __rshift__=_append_math_op(op.rshift),  # rshift the tree leaves
        __sub__=_append_math_op(op.sub),  # - the tree leaves
        __rsub__=_append_math_op(op.sub),  # - the tree leaves
        __truediv__=_append_math_op(op.truediv),  # / the tree leaves
        __xor__=_append_math_op(op.xor),  # xor the tree leaves
        tree_flatten=_flatten,  # jax.tree_util.tree_flatten rule
        tree_unflatten=classmethod(_unflatten),  # jax.tree_util.tree_unflatten rule
        at=property(_at_indexer),  # indexer to access a node in the tree
    )

    dcls = type(cls.__name__, (dcls,), attrs)
    return jtu.register_pytree_node_class(dcls)


def is_treeclass_equal(lhs, rhs):
    """Assert if two treeclasses are equal"""
    lhs_leaves, lhs_treedef = jtu.tree_flatten(lhs)
    rhs_leaves, rhs_treedef = jtu.tree_flatten(rhs)

    def is_node_equal(lhs_node, rhs_node):
        if isinstance(lhs_node, jnp.ndarray) and isinstance(rhs_node, jnp.ndarray):
            return jnp.array_equal(lhs_node, rhs_node)
        return lhs_node == rhs_node

    return (lhs_treedef == rhs_treedef) and all(
        [is_node_equal(lhs_leaves[i], rhs_leaves[i]) for i in range(len(lhs_leaves))]
    )


def is_nondiff(item: Any) -> bool:
    """Check if a node is non-differentiable."""

    def _is_nondiff_item(node: Any):
        if isinstance(node, (float, complex)) or dataclasses.is_dataclass(node):
            return False
        elif isinstance(node, jnp.ndarray) and jnp.issubdtype(node.dtype, jnp.inexact):
            return False

        return True

    if isinstance(item, Iterable):
        # if an iterable has at least one non-differentiable item
        # then the whole iterable is non-differentiable
        return any([_is_nondiff_item(item) for item in jtu.tree_leaves(item)])
    return _is_nondiff_item(item)


def tree_filter(tree: PyTree, *, where: Callable[[Any], bool] | PyTree = None):
    """Filter a tree based on a callable function or a pytree of booleans.

    Args:
        tree: The tree to filter.
        where: A callable function or a pytree of booleans. Defaults to filtering non-differentiable nodes.
    """
    return dcu.dataclass_filter(tree, where=(where or is_nondiff))


def tree_unfilter(tree: PyTree, *, where: Callable[[Any], bool] | PyTree = None):
    """Unfilter a tree based on a callable function or a pytree of booleans.

    Args:
        tree: The tree to unfilter.
        where: A callable function or a pytree of booleans. Defaults to unfiltering all nodes.
    """
    return dcu.dataclass_unfilter(tree, where=(where or (lambda _: True)))
