from dataclasses import dataclass, field
from typing import Any

import jax.tree_util as jtu

from pytreeclass.src.tree_base import treeBase


class ImmutableInstanceError(Exception):
    pass


@jtu.register_pytree_node_class
@dataclass(repr=False, eq=False, frozen=True)
class static(treeBase):
    value: Any = field(metadata={"static": True})
    __static_pytree__ = True


def static_value(value):
    return static(value)


def static_field(**kwargs):
    """ignore from pytree computations"""
    return field(**{**kwargs, **{"metadata": {"static": True}}})


class mutableContext:
    """Allow mutable behvior within this context"""

    def __init__(self, instance):
        assert hasattr(
            instance, "__immutable_treeclass__"
        ), "instance must be immutable treeclass"
        self.instance = instance

    def __enter__(self):
        object.__setattr__(self.instance, "__immutable_treeclass__", False)

    def __exit__(self, type_, value, traceback):
        object.__setattr__(self.instance, "__immutable_treeclass__", True)
