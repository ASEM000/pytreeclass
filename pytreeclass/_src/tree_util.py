from __future__ import annotations

import functools as ft
from collections.abc import Iterable
from dataclasses import Field, field
from types import FunctionType, MappingProxyType
from typing import Any, Callable

import jax.numpy as jnp
import jax.tree_util as jtu

import pytreeclass as pytc
import pytreeclass._src as src

PyTree = Any


def tree_copy(tree):
    return jtu.tree_unflatten(*jtu.tree_flatten(tree)[::-1])


def _tree_mutate(tree):
    """Enable mutable behavior for a treeclass instance"""
    if pytc.is_treeclass(tree):
        object.__setattr__(tree, "__immutable_pytree__", False)
        for field_item in pytc.fields(tree):
            if hasattr(tree, field_item.name):
                _tree_mutate(getattr(tree, field_item.name))
    return tree


def _tree_immutate(tree):
    """Enable immutable behavior for a treeclass instance"""
    if pytc.is_treeclass(tree):
        object.__setattr__(tree, "__immutable_pytree__", True)
        for field_item in pytc.fields(tree):
            if hasattr(tree, field_item.name):
                _tree_immutate(getattr(tree, field_item.name))
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
    ), f"`mutable` can only be applied to methods. Found{type(func)}"

    @ft.wraps(func)
    def mutable_method(self, *args, **kwargs):
        self = _tree_mutate(tree=self)
        output = func(self, *args, **kwargs)
        self = _tree_immutate(tree=self)
        return output

    return mutable_method


# filtering nondifferentiable fields


def _is_nondiff(item: Any) -> bool:
    """check if tree is non-differentiable"""

    def _is_nondiff_item(node: Any):
        """check if node is non-differentiable"""
        # differentiable types
        if isinstance(node, (float, complex, src.tree_base._treeBase)):
            return False

        # differentiable array
        elif isinstance(node, jnp.ndarray) and jnp.issubdtype(node.dtype, jnp.inexact):
            return False

        return True

    if isinstance(item, Iterable):
        # if an iterable has at least one non-differentiable item
        # then the whole iterable is non-differentiable
        return any([_is_nondiff_item(item) for item in jtu.tree_leaves(item)])
    return _is_nondiff_item(item)


def _append_field(
    tree: PyTree,
    where: Callable[[Field, Any], bool] | PyTree,
    replacing_field: Field = field,
) -> PyTree:
    """append a dataclass field to a treeclass `__undeclared_fields__`

    Args:
        tree (PyTree): tree to append field to
        where (Callable[[Field, Any], bool] | PyTree, optional): where to append field. Defaults to _is_nondiff.
        replacing_field (Field, optional): type of field. Defaults to field.

    Note:
        This is the base mechanism for controlling the static/dynamic behavior of a treeclass instance.

        during the `tree_flatten`, tree_fields are the combination of
        {__dataclass_fields__, __undeclared_fields__} this means that a field defined
        in `__undeclared_fields__` with the same name as in __dataclass_fields__
        will override its properties, this is useful if you want to change the metadata
        of a field but don't want to change the original field definition defined in the class.
    """

    def _callable_map(tree: PyTree, where: Callable[[Field, Any], bool]) -> PyTree:
        # filter based on a conditional callable
        for field_item in pytc.fields(tree):
            node_item = getattr(tree, field_item.name)

            if pytc.is_treeclass(node_item):
                _callable_map(tree=node_item, where=where)

            elif where(node_item):
                new_field = replacing_field(repr=field_item.repr)
                object.__setattr__(new_field, "name", field_item.name)
                object.__setattr__(new_field, "type", field_item.type)
                new_fields = {**tree.__undeclared_fields__, **{field_item.name: new_field}}  # fmt: skip
                object.__setattr__(tree, "__undeclared_fields__", MappingProxyType(new_fields))  # fmt: skip

        return tree

    def _mask_map(tree: PyTree, where: PyTree) -> PyTree:
        # filter based on a mask of the same type as `tree`
        for (lhs_field_item, rhs_field_item) in zip(
            pytc.fields(tree), pytc.fields(where)
        ):
            lhs_node_item = getattr(tree, lhs_field_item.name)
            rhs_node_item = getattr(where, rhs_field_item.name)

            if pytc.is_treeclass(lhs_node_item):
                _mask_map(tree=lhs_node_item, where=rhs_node_item)

            elif jnp.all(rhs_node_item):
                new_field = replacing_field(repr=lhs_field_item.repr)
                object.__setattr__(new_field, "name", lhs_field_item.name)
                object.__setattr__(new_field, "type", lhs_field_item.type)
                new_fields = {**tree.__undeclared_fields__, **{lhs_field_item.name: new_field}}  # fmt: skip
                object.__setattr__(tree, "__undeclared_fields__", MappingProxyType(new_fields))  # fmt: skip

        return tree

    if isinstance(where, FunctionType):
        return _callable_map(tree_copy(tree), where)
    elif isinstance(where, type(tree)):
        return _mask_map(tree_copy(tree), where)
    raise TypeError(f"`where` must be a Callable or a {type(tree)}")


def _unappend_field(tree: PyTree, cond: Callable[[Field], bool]) -> PyTree:
    """remove a dataclass field from `__undeclared_fields__` added if some condition is met"""

    def _recurse(tree):
        for field_item in pytc.fields(tree):
            node_item = getattr(tree, field_item.name)
            if pytc.is_treeclass(node_item):
                _recurse(tree=node_item)
            elif cond(field_item):
                new_fields = dict(tree.__undeclared_fields__)
                new_fields.pop(field_item.name)
                object.__setattr__(tree, "__undeclared_fields__", MappingProxyType(new_fields))  # fmt: skip
        return tree

    return _recurse(tree_copy(tree))


def filter_nondiff(
    tree: PyTree, where: PyTree | Callable[[Field, Any], bool] = None
) -> PyTree:
    def _is_nondiff_item(node: Any):
        if isinstance(node, (float, complex, src.tree_base._treeBase)):
            # differentiable types
            return False

        elif isinstance(node, jnp.ndarray) and jnp.issubdtype(node.dtype, jnp.inexact):
            # differentiable array
            return False

        return True

    def _is_nondiff(item: Any) -> bool:
        if isinstance(item, Iterable):
            # if an iterable has at least one non-differentiable item
            # then the whole iterable is non-differentiable
            return any([_is_nondiff_item(item) for item in jtu.tree_leaves(item)])
        return _is_nondiff_item(item)

    nondiff_field = ft.partial(pytc.field, nondiff=True)
    where = where or _is_nondiff
    return _append_field(tree=tree, where=where, replacing_field=nondiff_field)


def unfilter_nondiff(tree):
    return _unappend_field(tree, pytc.is_nondiff_field)


def tree_freeze(
    tree: PyTree, where: PyTree | Callable[[Field, Any], bool] = lambda _: True
) -> PyTree:

    frozen_field = ft.partial(pytc.field, frozen=True)

    return _append_field(tree=tree, where=where, replacing_field=frozen_field)


def tree_unfreeze(tree):
    return _unappend_field(tree, pytc.is_frozen_field)
