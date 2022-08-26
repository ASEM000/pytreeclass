from dataclasses import dataclass, field
from typing import Any

import jax.tree_util as jtu

import pytreeclass.src as src
from pytreeclass.src.tree_base import _treeBase


class ImmutableInstanceError(Exception):
    pass


@jtu.register_pytree_node_class
@dataclass(repr=False, eq=False, frozen=True)
class static(_treeBase):
    value: Any = field(metadata={"static": True})


def static_value(value):
    return static(value)


def static_field(**kwargs):
    """ignore from pytree computations"""
    return field(**{**kwargs, **{"metadata": {"static": True}}})


def mutate_tree(tree):
    if src.tree_util.is_treeclass(tree):
        object.__setattr__(tree, "__immutable_treeclass__", False)
        for field_item in tree.__pytree_fields__.values():
            if hasattr(tree, field_item.name):
                mutate_tree(getattr(tree, field_item.name))


def immutate_tree(tree):
    if src.tree_util.is_treeclass(tree):
        object.__setattr__(tree, "__immutable_treeclass__", True)
        for field_item in tree.__pytree_fields__.values():
            if hasattr(tree, field_item.name):
                immutate_tree(getattr(tree, field_item.name))


class mutableContext:
    """Allow mutable behvior within this context"""

    def __init__(self, instance):
        assert hasattr(
            instance, "__immutable_treeclass__"
        ), "instance must be immutable treeclass"
        self.instance = instance

    def __enter__(self):
        mutate_tree(self.instance)

    def __exit__(self, type_, value, traceback):
        immutate_tree(self.instance)
