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

# def _filter_static_field(field):
# return field.metadata.get("static", False)


class _fieldDict(dict):
    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class _treeBase:
    def __new__(cls, *args, **kwargs):
        self = object.__new__(cls)

        object.__setattr__(self, "__undeclared_fields__", {})
        object.__setattr__(self, "__nondiff_keys__", [])
        object.__setattr__(self, "__inexact_pytree__", False)

        for field_item in self.__dataclass_fields__.values():
            if field_item.default is not MISSING:
                object.__setattr__(self, field_item.name, field_item.default)

        return self

    def __repr__(self):
        return tree_repr(self, width=60)

    def __str__(self):
        return tree_str(self, width=60)

    def __hash__(self):
        return hash(tuple(jtu.tree_leaves(self)))

    @property
    def frozen(self) -> bool:
        """Show treeclass frozen status.

        Returns:
            Frozen state boolean.
        """
        return True if hasattr(self, "__frozen_structure__") else False

    def summary(self, array: jnp.ndarray = None) -> str:
        return tree_summary(self, array)

    def tree_diagram(self) -> str:
        return tree_diagram(self)

    def tree_box(self, array: jnp.ndarray = None) -> str:
        return tree_box(self, array)

    @property
    def __pytree_structure__(self):
        if (
            hasattr(self, "__frozen_structure__")
            and getattr(self, "__frozen_structure__") is not None
        ):
            # check if frozen structure is cached
            # Note: frozen strcture = None
            # means that the tree is frozen, but not yet cached
            return self.__frozen_structure__

        dynamic, static = _fieldDict(), _fieldDict()

        for field_item in self.__pytree_fields__.values():
            if field_item.metadata.get("static", False):
                static[field_item.name] = getattr(self, field_item.name)
            else:
                dynamic[field_item.name] = getattr(self, field_item.name)

        static["__undeclared_fields__"] = self.__undeclared_fields__

        return (dynamic, static)

    def tree_flatten(self):
        """Flatten rule for `jax.tree_flatten`

        Returns:
            Tuple of dynamic values and (dynamic keys,static dict, cached values)
        """
        dynamic, static = self.__pytree_structure__

        if self.frozen:
            return (), ((), (), (dynamic, static))

        else:
            return dynamic.values(), (dynamic.keys(), (static), ())

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

        self = object.__new__(cls)

        if len(treedef[2]) > 0:
            dynamic, static = treedef[2]
            # retrieve the cached frozen structure
            object.__setattr__(self, "__frozen_structure__", (dynamic, static))
        else:
            dynamic = dict(zip(treedef[0], leaves))
            static = treedef[1]

        self.__dict__.update(dynamic)
        self.__dict__.update(static)

        return self

    def register_node(
        self, node: Any, *, name: str, static: bool = False, repr: bool = True
    ) -> Any:
        """Add item to dataclass fields to bee seen by jax computations"""
        if hasattr(self, name) and (name in self.__undeclared_fields__):
            return getattr(self, name)

        # create field
        field_value = field(repr=repr, metadata={"static": static})

        object.__setattr__(field_value, "name", name)
        object.__setattr__(field_value, "type", type(node))

        # register it to class
        self.__undeclared_fields__.update({name: field_value})
        object.__setattr__(self, name, node)

        return getattr(self, name)

    @property
    def __pytree_fields__(self):
        if len(self.__undeclared_fields__) == 0:
            return self.__dataclass_fields__
        else:
            return {**self.__dataclass_fields__, **self.__undeclared_fields__}


class _implicitTreeBase:
    """Register dataclass fields and treeclass instance variables"""

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

        if (is_treeclass(value)) and (name not in self.__pytree_fields__):
            # create field
            field_value = field()

            object.__setattr__(field_value, "name", name)
            object.__setattr__(field_value, "type", type(value))

            # register it to class
            self.__undeclared_fields__.update({name: field_value})
