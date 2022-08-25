from __future__ import annotations

from dataclasses import MISSING, field
from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu

from pytreeclass.src.tree_util import is_treeclass, static_value
from pytreeclass.src.tree_viz import (
    tree_box,
    tree_diagram,
    tree_repr,
    tree_str,
    tree_summary,
)

PyTree = Any


class fieldDict(dict):
    # dict will throw
    # `ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()``
    # while this implementation will not.
    # relevant issue : https://github.com/google/jax/issues/11089
    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class treeBase:
    def __new__(cls, *args, **kwargs):

        self = super().__new__(cls)

        object.__setattr__(self, "__treeclass_fields__", cls.__dataclass_fields__)

        # register dataclass fields to instance dict
        # otherwise will raise undeclared error for non defined
        # init classes.
        for field_item in self.__treeclass_fields__.values():
            if field_item.default is not MISSING:
                object.__setattr__(self, field_item.name, field_item.default)
        return self

    @property
    def frozen(self) -> bool:
        """Show treeclass frozen status.

        Returns:
            Frozen state boolean.
        """
        return True if hasattr(self, "__frozen_fields__") else False

    def tree_flatten(self):
        """Flatten rule for `jax.tree_flatten`

        Returns:
            Tuple of dynamic values and (dynamic keys,static dict)
        """
        dynamic, static = self.__treeclass_structure__

        if self.frozen:
            return (), ((), {"__frozen_fields__": (dynamic, static)})

        else:
            return dynamic.values(), (dynamic.keys(), static)

    @classmethod
    def tree_unflatten(cls, treedef, children):
        """Unflatten rule for `jax.tree_unflatten`

        Args:
            treedef:
                Pytree definition
                includes Dynamic nodes keys , static dictionary and frozen state
            children:
                Dynamic nodes values

        Returns:
            New class instance
        """

        new_cls = object.__new__(cls)
        tree_fields = treedef[1].get("__frozen_fields__", None)

        if tree_fields is not None:
            object.__setattr__(new_cls, "__frozen_fields__", tree_fields)
            dynamic, static = tree_fields
            attrs = {**dynamic, **static}

        else:
            attrs = {**dict(zip(treedef[0], children)), **treedef[1]}

        new_cls.__dict__.update(attrs)

        return new_cls

    def __hash__(self):
        return hash(tuple(jtu.tree_leaves(self)))

    def asdict(self) -> dict[str, Any]:
        """Dictionary representation of dataclass_fields"""
        dynamic, static = self.__treeclass_structure__
        static.pop("__treeclass_fields__", None)
        static.pop("__immutable_treeclass__", None)
        return {
            **dynamic,
            **jtu.tree_map(
                lambda x: x.value if isinstance(x, static_value) else x, dict(static)
            ),
        }

    def register_node(
        self, node: Any, *, name: str, static: bool = False, repr: bool = True
    ) -> Any:
        """Add item to dataclass fields to bee seen by jax computations"""
        if hasattr(self, name) and (name in self.__treeclass_fields__):
            return getattr(self, name)

        # create field
        field_value = field(repr=repr, metadata={"static": static})

        object.__setattr__(field_value, "name", name)
        object.__setattr__(field_value, "type", type(node))

        # register it to class
        self.__treeclass_fields__.update({name: field_value})
        object.__setattr__(self, name, node)

        return getattr(self, name)

    def __repr__(self):
        return tree_repr(self)

    def __str__(self):
        return tree_str(self)

    def summary(self, array: jnp.ndarray = None) -> str:
        return tree_summary(self, array)

    def tree_diagram(self) -> str:
        return tree_diagram(self)

    def tree_box(self, array: jnp.ndarray = None) -> str:
        return tree_box(self, array)

    @property
    def __treeclass_structure__(self):
        """Computes the dynamic and static fields.

        Returns:
            Pair of dynamic and static dictionaries.
        """
        if self.__dict__.get("__frozen_fields__", None) is not None:
            return self.__frozen_fields__

        dynamic, static = fieldDict(), fieldDict()

        for fi in self.__treeclass_fields__.values():
            # field value is defined in class dict
            if hasattr(self, fi.name):
                value = getattr(self, fi.name)
            else:
                # the user did not declare a variable defined in field
                raise ValueError(f"field={fi.name} is not declared.")

            if fi.metadata.get("static", False) or isinstance(value, static_value):
                static[fi.name] = value

            else:
                dynamic[fi.name] = value

        static["__treeclass_fields__"] = self.__treeclass_fields__

        return (dynamic, static)


class implicitTreeBase:
    """Register dataclass fields and treeclass instance variables"""

    def __setattr__(self, name: str, value: Any) -> None:

        if (is_treeclass(value)) and (name not in self.__treeclass_fields__):
            # create field
            field_value = field(repr=repr)

            object.__setattr__(field_value, "name", name)
            object.__setattr__(field_value, "type", type(value))

            # register it to class
            self.__treeclass_fields__.update({name: field_value})

        object.__setattr__(self, name, value)
