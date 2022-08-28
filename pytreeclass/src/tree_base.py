from __future__ import annotations

from dataclasses import MISSING, field
from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu

# from pytreeclass.src.decorator import static_value
from pytreeclass.src.tree_util import is_treeclass
from pytreeclass.src.tree_viz import (
    tree_box,
    tree_diagram,
    tree_repr,
    tree_str,
    tree_summary,
)

PyTree = Any


@jtu.register_pytree_node_class
class _STATIC:
    # exclude from JAX computations
    def __init__(self, value):
        self.value = value

    def tree_flatten(self):
        return (), (self.value)

    @classmethod
    def tree_unflatten(cls, treedef, leaves):
        return treedef

    def __repr__(self):
        return f"{self.value!r}"

    def __str__(self):
        return f"{self.value!s}"


class _nodeContainer(tuple):
    # tuple will throw
    # `ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()``
    # while this implementation will not.
    # relevant issue : https://github.com/google/jax/issues/11089
    def __init__(self, node_items):
        super().__init__()
        self.node_items = node_items

    def __eq__(self, other):
        return self.node_items == other.node_items


class _treeBase:
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        object.__setattr__(self, "__pytree_fields__", cls.__dataclass_fields__)
        # register dataclass fields to instance dict
        # otherwise will raise undeclared error for non defined
        # init classes.
        for field_item in self.__pytree_fields__.values():
            if field_item.default is not MISSING:
                object.__setattr__(self, field_item.name, field_item.default)
        return self

    @property
    def frozen(self) -> bool:
        """Show treeclass frozen status.

        Returns:
            Frozen state boolean.
        """
        return True if hasattr(self, "__frozen_structure__") else False

    def tree_flatten(self):
        """Flatten rule for `jax.tree_flatten`

        Returns:
            Tuple of dynamic values and (dynamic keys,static dict, cached values)
        """
        node_items, pytree_fields = self.__pytree_structure__

        if self.frozen:
            return (), ((), (node_items, pytree_fields))

        else:
            return node_items, (pytree_fields, ())

    @classmethod
    def tree_unflatten(cls, treedef, leaves):
        """Unflatten rule for `jax.tree_unflatten`

        Args:
            treedef:
                Pytree definition
                includes Dynamic nodes keys , static dictionary and frozen state
            leaves:
                Dynamic nodes values

        Returns:
            New class instance
        """

        new_cls = object.__new__(cls)

        if len(treedef[1]) > 0:
            node_items, field_items = treedef[1]
            # retrieve the cached frozen structure
            object.__setattr__(
                new_cls, "__frozen_structure__", (node_items, field_items)
            )
        else:
            node_items, field_items = leaves, treedef[0]

        attrs = dict(zip(field_items.keys(), node_items))
        new_cls.__dict__.update(attrs)
        new_cls.__dict__.update({"__pytree_fields__": (field_items)})

        return new_cls

    def __hash__(self):
        return hash(tuple(jtu.tree_leaves(self)))

    def register_node(
        self, node: Any, *, name: str, static: bool = False, repr: bool = True
    ) -> Any:
        """Add item to dataclass fields to bee seen by jax computations"""
        if hasattr(self, name) and (name in self.__pytree_fields__):
            return getattr(self, name)

        # create field
        field_value = field(repr=repr, metadata={"static": static})

        object.__setattr__(field_value, "name", name)
        object.__setattr__(field_value, "type", type(node))

        # register it to class
        self.__pytree_fields__.update({name: field_value})
        object.__setattr__(self, name, node)

        return getattr(self, name)

    def __repr__(self):
        return tree_repr(self, width=60)

    def __str__(self):
        return tree_str(self, width=60)

    def summary(self, array: jnp.ndarray = None) -> str:
        return tree_summary(self, array)

    def tree_diagram(self) -> str:
        return tree_diagram(self)

    def tree_box(self, array: jnp.ndarray = None) -> str:
        return tree_box(self, array)

    @property
    def __pytree_structure__(self):
        if self.__dict__.get("__frozen_structure__", None) is not None:
            # check if frozen structure is cached
            # Note: frozen strcture = None
            # means that the tree is frozen, but not yet cached
            return self.__frozen_structure__

        node_items = _nodeContainer(
            _STATIC(getattr(self, fi.name))
            if fi.metadata.get("static", False)
            else getattr(self, fi.name)
            for fi in self.__pytree_fields__.values()
        )

        return (node_items, self.__pytree_fields__)


class _implicitSetter:
    """Register dataclass fields and treeclass instance variables"""

    def __setattr__(self, name: str, value: Any) -> None:

        if (is_treeclass(value)) and (name not in self.__pytree_fields__):
            # create field
            field_value = field()

            object.__setattr__(field_value, "name", name)
            object.__setattr__(field_value, "type", type(value))

            # register it to class
            self.__pytree_fields__.update({name: field_value})

        object.__setattr__(self, name, value)
