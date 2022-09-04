from __future__ import annotations

import functools as ft
import sys
from dataclasses import field
from types import FunctionType
from typing import Any, Callable

import jax.numpy as jnp
import jax.tree_util as jtu

import pytreeclass.src as src
from pytreeclass.src.decorator_util import dispatch
from pytreeclass.src.tree_util import _pytree_map, is_treeclass


def static_field(**kwargs):
    """ignore from pytree computations"""
    return field(**{**kwargs, **{"metadata": {"static": True}}})


def _mutate_tree(tree):
    """Enable mutable behavior for a treeclass instance"""
    if src.tree_util.is_treeclass(tree):
        object.__setattr__(tree, "__immutable_pytree__", False)
        for field_item in tree.__pytree_fields__.values():
            if hasattr(tree, field_item.name):
                _mutate_tree(getattr(tree, field_item.name))
    return tree


def _immutate_tree(tree):
    """Enable immutable behavior for a treeclass instance"""
    if src.tree_util.is_treeclass(tree):
        object.__setattr__(tree, "__immutable_pytree__", True)
        for field_item in tree.__pytree_fields__.values():
            if hasattr(tree, field_item.name):
                _immutate_tree(getattr(tree, field_item.name))
    return tree


def _mutable(func):
    """decorator that allow mutable behvior
    for class methods/ function with treeclass as first argument

    Example:
        class ... :

        >>> @_mutable
        ... def mutable_method(self):
        ...    return self.value + 1
    """
    assert isinstance(
        func, FunctionType
    ), f"mutable can only be applied to methods. Found{type(func)}"

    @ft.wraps(func)
    def mutable_method(self, *args, **kwargs):
        self = _mutate_tree(tree=self)
        output = func(self, *args, **kwargs)
        self = _immutate_tree(tree=self)
        return output

    return mutable_method


def _freeze_nodes(tree):
    """inplace freezing"""
    # cache the tree structure (dynamic/static)
    if is_treeclass(tree):
        object.__setattr__(tree, "__frozen_structure__", tree.__pytree_structure__)
        for kw in tree.__pytree_fields__:
            _freeze_nodes(tree.__dict__[kw])
    return tree


def _unfreeze_nodes(tree):
    """inplace unfreezing"""
    # remove the cached frozen structure
    if is_treeclass(tree):
        if hasattr(tree, "__frozen_structure__"):
            object.__delattr__(tree, "__frozen_structure__")
        for kw in tree.__pytree_fields__:
            _unfreeze_nodes(tree.__dict__[kw])
    return tree


def _reduce_count_and_size(leaf):
    """reduce params count and params size of a tree of leaves"""

    def reduce_func(acc, node):
        lhs_count, lhs_size = acc
        rhs_count, rhs_size = _node_count_and_size(node)
        return (lhs_count + rhs_count, lhs_size + rhs_size)

    return jtu.tree_reduce(reduce_func, leaf, (complex(0, 0), complex(0, 0)))


def _node_count_and_size(node: Any) -> tuple[complex, complex]:
    """Calculate number and size of `trainable` and `non-trainable` parameters

    Args:
        node (Any): treeclass node

    Returns:
        complex: Complex number of (inexact, exact) parameters for count/size
    """

    @dispatch(argnum=0)
    def count_and_size(node):
        count = complex(0, 0)
        size = complex(0, 0)
        return count, size

    @count_and_size.register(jnp.ndarray)
    def _(node):
        # inexact(trainable) array
        if jnp.issubdtype(node, jnp.inexact):
            count = complex(int(jnp.array(node.shape).prod()), 0)
            size = complex(int(node.nbytes), 0)

        # exact paramter
        else:
            count = complex(0, int(jnp.array(node.shape).prod()))
            size = complex(0, int(node.nbytes))
        return count, size

    @count_and_size.register(float)
    @count_and_size.register(complex)
    def _(node):
        # inexact non-array (array_like)
        count = complex(1, 0)
        size = complex(sys.getsizeof(node), 0)
        return count, size

    @count_and_size.register(int)
    def _(node):
        # exact non-array
        count = complex(0, 1)
        size = complex(0, sys.getsizeof(node))
        return count, size

    return count_and_size(node)


def _copy_field(
    field_item,
    *,
    field_name: str = None,
    field_type: type = None,
    compare: bool = None,
    default: Any = None,
    default_factory: Callable = None,
    hash: Callable = None,
    init: bool = None,
    repr: bool = None,
    metadata: dict[str, Any] = None,
    aux_metadata: dict[str, Any] = None,
):
    """copy a field with new values"""
    # creation of a new field avoid mutating the original field
    aux_metadata = aux_metadata or {}

    new_field = field(
        compare=compare or getattr(field_item, "compare"),
        default=default or getattr(field_item, "default"),
        default_factory=default_factory or getattr(field_item, "default_factory"),
        hash=hash or getattr(field_item, "hash"),
        init=init or getattr(field_item, "init"),
        metadata=metadata
        or {
            **getattr(field_item, "metadata"),
            **aux_metadata,
        },
        repr=repr or getattr(field_item, "repr"),
    )

    object.__setattr__(new_field, "name", field_name or getattr(field_item, "name"))
    object.__setattr__(new_field, "type", field_type or getattr(field_item, "type"))

    return new_field


def _is_nondiff(node):
    """check if node is non-differentiable"""
    return isinstance(node, (int, bool, str)) or (
        isinstance(node, (jnp.ndarray)) and not jnp.issubdtype(node.dtype, jnp.inexact)
    )


def filter_nondiff(tree):
    """filter non-differentiable fields from a treeclass instance"""
    # we use _pytree_map to add {nondiff:True} to a non-differentiable field metadata
    # this operation is done in-place and changes the tree structure
    # thus its not bundled with `.at[..]` as it will break composability
    return _pytree_map(
        tree,
        cond=lambda _, __, node_item: _is_nondiff(node_item),
        true_func=lambda tree, field_item, node_item: {
            **tree.__undeclared_fields__,
            **{
                field_item.name: _copy_field(
                    field_item, aux_metadata={"static": True, "nondiff": True}
                )
            },
        },
        false_func=lambda tree, field_item, node_item: tree.__undeclared_fields__,
        attr_func=lambda _, __, ___: "__undeclared_fields__",
        is_leaf=lambda _, field_item, __: field_item.metadata.get("static", False),
    )


def unfilter_nondiff(tree):
    """remove fields added by `filter_nondiff"""
    return _pytree_map(
        tree,
        cond=lambda _, __, ___: True,
        true_func=lambda tree, field_item, node_item: {
            field_name: field_value
            for field_name, field_value in tree.__undeclared_fields__.items()
            if not field_value.metadata.get("nondiff", False)
        },
        false_func=lambda _, __, ___: {},
        attr_func=lambda _, __, ___: "__undeclared_fields__",
        is_leaf=lambda _, __, ___: False,
    )
