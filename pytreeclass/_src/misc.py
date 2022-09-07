from __future__ import annotations

import functools as ft
from dataclasses import field
from types import FunctionType
from typing import Any, Callable

import jax.numpy as jnp

from pytreeclass._src.tree_util import (
    _pytree_map,
    _tree_immutate,
    _tree_mutate,
    is_treeclass,
)


def static_field(**kwargs):
    """ignore from pytree computations"""
    return field(**{**kwargs, **{"metadata": {"static": True}}})


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
        self = _tree_mutate(tree=self)
        output = func(self, *args, **kwargs)
        self = _tree_immutate(tree=self)
        return output

    return mutable_method


def _add_temp_method(func, name: str, method: Callable[[Any]]):
    """function that allow to add a temporary method to a class."""

    def wrapper(self, *args, **kwargs):
        # Add the method to the class before calling it
        object.__setattr__(self, name, ft.partial(method, self))
        output = func(self, *args, **kwargs)
        # Remove the method from the class after calling it
        object.__delattr__(self, name)
        return output

    return wrapper


def param(
    tree, node: Any, *, name: str, static: bool = False, repr: bool = True
) -> Any:
    """Add and return a parameter to the treeclass in a compact way.

    Note:
        If the node is already defined (checks by name) then it will be returned
        Useful if node definition

    Args:
        node (Any): Any node to be added to the treeclass
        name (str): Name of the node
        static (bool, optional): Whether to exclude from tree leaves. Defaults to False.
        repr (bool, optional): whether to show in repr/str/tree_viz . Defaults to True.


    Example:
        @pytc.treeclass
        class StackedLinear:

        def __init__(self,key):
            self.keys = jax.random.split(key,3)

        def __call__(self,x):
            x = self.param(... ,name="l1")(x)
            return x
    """
    if not is_treeclass(tree):
        raise TypeError(f"param can only be applied to treeclass. Found {type(tree)}")

    if hasattr(tree, name) and (name in tree.__undeclared_fields__):
        return getattr(tree, name)

    # create field
    field_value = field(repr=repr, metadata={"static": static, "param": True})

    object.__setattr__(field_value, "name", name)
    object.__setattr__(field_value, "type", type(node))

    # register it to class
    tree.__undeclared_fields__.update({name: field_value})
    object.__setattr__(tree, name, node)

    return getattr(tree, name)


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
