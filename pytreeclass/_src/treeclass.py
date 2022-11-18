from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import pytreeclass._src.dataclass_util as dcu
from pytreeclass._src.tree_indexer import _treeIndexer
from pytreeclass._src.tree_op import _treeOp
from pytreeclass._src.tree_util import _fieldDict, _mutable, _tree_structure
from pytreeclass.tree_viz.tree_pprint import tree_repr, tree_str


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


def _flatten(tree) -> tuple[Any, tuple[str, _fieldDict[str, Any]]]:
    """Flatten rule for `jax.tree_flatten`"""
    dynamic, static = _tree_structure(tree)
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
        tree_flatten=_flatten,  # jax.tree_util.tree_flatten rule
        tree_unflatten=classmethod(_unflatten),  # jax.tree_util.tree_unflatten rule
    )

    dcls = type(cls.__name__, (dcls, _treeIndexer, _treeOp), attrs)
    return jax.tree_util.register_pytree_node_class(dcls)


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
