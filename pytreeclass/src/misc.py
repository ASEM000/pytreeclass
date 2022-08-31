import functools as ft
from dataclasses import field
from types import FunctionType
from typing import Any, Callable

import jax.numpy as jnp

import pytreeclass.src as src
from pytreeclass.src.tree_util import is_treeclass, tree_copy


class ImmutableInstanceError(Exception):
    pass


def static_field(**kwargs):
    """ignore from pytree computations"""
    return field(**{**kwargs, **{"metadata": {"static": True}}})


def _mutate_tree(tree):
    if src.tree_util.is_treeclass(tree):
        object.__setattr__(tree, "__immutable_treeclass__", False)
        for field_item in tree.__pytree_fields__.values():
            if hasattr(tree, field_item.name):
                _mutate_tree(getattr(tree, field_item.name))
    return tree


def _immutate_tree(tree):
    if src.tree_util.is_treeclass(tree):
        object.__setattr__(tree, "__immutable_treeclass__", True)
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
        return func(self, *args, **kwargs)

    return mutable_method


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
    metadata: dict = None,
    aux_metadata: dict = None,
):

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
    return isinstance(node, (int, bool, str)) or (
        isinstance(node, (jnp.ndarray)) and not jnp.issubdtype(node.dtype, jnp.inexact)
    )


def unfilter_nondiff(tree):
    # unfilter nondiff nodes inplace
    # by removing nondiff_fields from the undeclared_fields variable

    def recurse(tree):
        object.__setattr__(
            tree,
            "__undeclared_fields__",
            {
                field_name: field_value
                for field_name, field_value in tree.__undeclared_fields__.items()
                if not field_value.metadata.get("nondiff", False)
            },
        )

        for field_item in tree.__pytree_fields__.values():
            field_value = getattr(tree, field_item.name)
            if is_treeclass(field_value):
                recurse(field_value)
        return tree

    return recurse(tree_copy(tree)) if is_treeclass(tree) else tree


def filter_nondiff(tree):
    # filter nondiff nodes inplace
    # to be used under jax.grad transformation
    def recurse(tree):
        for field_item in tree.__pytree_fields__.values():
            field_value = getattr(tree, field_item.name)

            if not field_item.metadata.get("static", False):
                if is_treeclass(field_value):
                    recurse(field_value)

                elif _is_nondiff(field_value):
                    new_field = _copy_field(
                        field_item=field_item,
                        aux_metadata={"nondiff": True, "static": True},
                    )

                    tree.__undeclared_fields__.update({field_item.name: new_field})
        return tree

    return recurse(tree_copy(tree)) if is_treeclass(tree) else tree
