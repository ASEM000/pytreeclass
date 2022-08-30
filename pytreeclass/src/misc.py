import functools as ft
from dataclasses import dataclass, field
from types import FunctionType
from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu

import pytreeclass.src as src
from pytreeclass.src.tree_base import _treeBase


class ImmutableInstanceError(Exception):
    pass


@jtu.register_pytree_node_class
@dataclass(repr=False, eq=True, frozen=True)
class static(_treeBase):
    value: Any = field(metadata={"static": True})


def static_value(value):
    return static(value)


def static_field(**kwargs):
    """ignore from pytree computations"""
    return field(**{**kwargs, **{"metadata": {"static": True}}})


def _mutate_tree(tree):
    if src.tree_util.is_treeclass(tree):
        object.__setattr__(tree, "__immutable_treeclass__", False)
        for field_item in tree.__pytree_fields__.values():
            if hasattr(tree, field_item.name):
                _mutate_tree(getattr(tree, field_item.name))


def _immutate_tree(tree):
    if src.tree_util.is_treeclass(tree):
        object.__setattr__(tree, "__immutable_treeclass__", True)
        for field_item in tree.__pytree_fields__.values():
            if hasattr(tree, field_item.name):
                _immutate_tree(getattr(tree, field_item.name))


@dataclass(eq=False, frozen=True)
class _mutableContext:
    """Allow mutable behvior within this context"""

    instance: Any

    def __post_init__(self):
        assert hasattr(
            self.instance, "__immutable_treeclass__"
        ), "instance must be immutable treeclass"

    def __enter__(self):
        _mutate_tree(self.instance)

    def __exit__(self, type_, value, traceback):
        _immutate_tree(self.instance)


def _mutable(instance_method):
    """decorator that allow mutable behvior"""
    assert isinstance(
        instance_method, FunctionType
    ), f"mutable can only be applied to methods. Found{type(instance_method)}"

    @ft.wraps(instance_method)
    def mutable_method(self, *args, **kwargs):
        with _mutableContext(self):
            # return before exiting the context
            # will lead to mutable behavior
            return instance_method(self, *args, **kwargs)

    return mutable_method


def _unfilter_nondiff(tree):
    # unfilter nondiff nodes inplace
    # by removing aux_fields from undeclared_fields

    def recurse(tree):
        undeclared_fields = tree.__undeclared_fields__
        undeclared_fields = {
            field_name: field_value
            for field_name, field_value in undeclared_fields.items()
            if not field_value.metadata.get("nondiff", False)
        }
        object.__setattr__(tree, "__undeclared_fields__", undeclared_fields)

        for field_item in tree.__pytree_fields__.values():
            field_value = getattr(tree, field_item.name)
            if hasattr(field_value, "__dataclass_fields__"):
                recurse(field_value)
        return tree

    return recurse(tree)


def _filter_nondiff(tree):
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


@dataclass(eq=False, frozen=True)
class diffContext:
    instance: Any

    def __post_init__(self):
        assert hasattr(
            self.instance, "__immutable_treeclass__"
        ), "instance must be immutable treeclass"

    def __enter__(self):
        _filter_nondiff(self.instance)

    def __exit__(self, *args):
        _unfilter_nondiff(self.instance)
