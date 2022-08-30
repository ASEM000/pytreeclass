import functools as ft
from dataclasses import dataclass, field
from types import FunctionType
from typing import Any

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
class mutableContext:
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
        with mutableContext(self):
            # return before exiting the context
            # will lead to mutable behavior
            return instance_method(self, *args, **kwargs)

    return mutable_method
