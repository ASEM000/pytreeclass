import functools as ft
from dataclasses import field
from types import FunctionType

import jax.numpy as jnp
import jaxlib

import pytreeclass.src as src
from pytreeclass.src.tree_util import tree_copy


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


def _unfilter_nondiff_fields(tree):
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
            if hasattr(field_value, "__dataclass_fields__"):
                recurse(field_value)
        return tree

    return recurse(tree)


def _filter_nondiff_fields(tree):
    # filter nondiff nodes inplace

    def modify_field_static(field_item, static: bool = True):
        new_field = field(
            compare=getattr(field_item, "compare"),
            default=getattr(field_item, "default"),
            default_factory=getattr(field_item, "default_factory"),
            hash=getattr(field_item, "hash"),
            init=getattr(field_item, "init"),
            metadata={
                **getattr(field_item, "metadata"),
                **{"static": static, "nondiff": True},
            },
            repr=getattr(field_item, "repr"),
        )

        object.__setattr__(new_field, "name", getattr(field_item, "name"))
        object.__setattr__(new_field, "type", getattr(field_item, "type"))

        return new_field

    def is_nondiff(node):
        return isinstance(node, (int, bool, str)) or (
            isinstance(node, (jnp.ndarray))
            and not jnp.issubdtype(node.dtype, jnp.inexact)
        )

    def recurse(tree):
        for field_item in tree.__pytree_fields__.values():
            field_value = getattr(tree, field_item.name)

            if not field_item.metadata.get("static", False):

                if hasattr(field_value, "__dataclass_fields__"):
                    recurse(field_value)

                elif is_nondiff(field_value):
                    new_field = modify_field_static(field_item, static=True)

                    object.__setattr__(
                        tree,
                        "__undeclared_fields__",
                        {
                            **getattr(tree, "__undeclared_fields__"),
                            **{new_field.name: new_field},
                        },
                    )

        return tree

    return recurse(tree)


def filter_nondiff(func):
    """decorator that sets nondiff fields to be static
    for class methods/ function with treeclass as first argument

    Example:

    @filter_nondiff
    @jax.jit
    @jax.value_and_grad
    def update(model, *args, **kwargs):
        ...
    """
    assert isinstance(
        func, (FunctionType, jaxlib.xla_extension.CompiledFunction)
    ), f"filter can only be applied to methods/Functions. Found{type(func)}"

    @ft.wraps(func)
    def filter_nondiff_method(self, *args, **kwargs):
        assert hasattr(
            self, "__immutable_treeclass__"
        ), "instance must be immutable treeclass"
        new_self = tree_copy(self)
        _filter_nondiff_fields(new_self)
        value = func(new_self, *args, **kwargs)
        _unfilter_nondiff_fields(new_self)
        return value

    return filter_nondiff_method
