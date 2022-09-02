from __future__ import annotations

from dataclasses import MISSING, field
from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu

# from pytreeclass.src.decorator import static_value
from pytreeclass.src.tree_viz import (
    tree_box,
    tree_diagram,
    tree_repr,
    tree_str,
    tree_summary,
)

PyTree = Any


class _fieldDict(dict):
    """A dict used for `__pytree_structure__` attribute of a treeclass instance"""

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class _treeBase:
    def __new__(cls, *args, **kwargs):
        self = object.__new__(cls)

        object.__setattr__(self, "__undeclared_fields__", {})

        # set default values to class instance
        # Note: ideally this method should be called once to avoid multiple
        # definition of `__undeclared_fields__` attribute
        for field_item in self.__dataclass_fields__.values():
            if field_item.default is not MISSING:
                object.__setattr__(self, field_item.name, field_item.default)

        return self

    def __repr__(self) -> str:
        """pretty print pytree instance"""
        return tree_repr(self, width=60)

    def __str__(self):
        """pretty print pytree instance"""
        return tree_str(self, width=60)

    def __hash__(self):
        """hash rule for pytree instance"""
        return hash(tuple(jtu.tree_leaves(self)))

    @property
    def frozen(self) -> bool:
        """Show treeclass frozen status.

        Returns:
            Frozen state boolean.
        """
        return True if hasattr(self, "__frozen_structure__") else False

    def summary(self, array: jnp.ndarray = None) -> str:
        """print a summary of the pytree instance

        Example:
            >>> import pytreeclass as pytc
            >>> @pytc.treeclass
            ... class A:
            ...     a: int = 1
            ...     b: float = 2.0
            >>> a = A()
            >>> print(a.summary())
            ┌────┬─────┬───────┬────────┬──────┐
            │Name│Type │Param #│Size    │Config│
            ├────┼─────┼───────┼────────┼──────┤
            │a   │int  │0(1)   │0.00B   │a=1   │
            │    │     │       │(28.00B)│      │
            ├────┼─────┼───────┼────────┼──────┤
            │b   │float│1(0)   │24.00B  │b=2.0 │
            │    │     │       │(0.00B) │      │
            └────┴─────┴───────┴────────┴──────┘
            Total count :   1(1)
            Dynamic count : 1(1)
            Frozen count :  0(0)
            ----------------------------------------
            Total size :    24.00B(28.00B)
            Dynamic size :  24.00B(28.00B)
            Frozen size :   0.00B(0.00B)
            ========================================
        """

        return tree_summary(self, array)

    def tree_diagram(self) -> str:
        """Print a diagram of the pytree instance

        Example:
            >>> import pytreeclass as pytc
            >>> @pytc.treeclass
            ... class A:
            ...     a: int = 1
            ...     b: float = 2.0
            >>> a = A()
            >>> print(a.tree_diagram())
            A
                ├── a=1
                └── b=2.0
        """
        return tree_diagram(self)

    def tree_box(self, array: jnp.ndarray = None) -> str:
        """keras-like `plot_model` subclassing paradigm"""
        return tree_box(self, array)

    @property
    def __pytree_structure__(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """return dynamic and static fields of the pytree instance"""
        if hasattr(self, "__frozen_structure__"):
            # check if pytree_structure is cached
            # ** another approach is to append {static:True} to the metadata using `_pytree_map`,
            # however this will be a bit slower as the tree_flatten has to traverse all fields
            # while here, no traversal is needed
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

        # using `object.__new__`` here is faster than using `cls.__new__`
        # as it avoids calling bases __new__ methods
        # moreover , in _treeBase.__new__ we declare `__undeclared_fields__`
        # however, using obejct we will not have this attribute,
        # so we need to add these attributes in the static, that updates the `self.__dict__``
        # since we already have to pass `__undeclared_fields__` through flatten/unflatten
        # this approach creates the attribute once.
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

    def param(
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
        """Return a dictionary of all fields in the dataclass"""
        # in case of explicit treebase with no `param` then
        # its preferable to create a new dict and just point to `__dataclass_fields__`
        return (
            self.__dataclass_fields__
            if len(self.__undeclared_fields__) == 0
            else {**self.__dataclass_fields__, **self.__undeclared_fields__}
        )


class _implicitSetter:
    """Register dataclass fields and treeclass instance variables"""

    __immutable_treeclass__ = True

    def __setattr__(self, key: str, value: Any) -> None:

        if self.__immutable_treeclass__:
            raise ImmutableInstanceError(
                f"Cannot set {key} = {value}. Use `.at['{key}'].set({value!r})` instead."
            )

        object.__setattr__(self, key, value)

        if (isinstance(value, _treeBase)) and (key not in self.__pytree_fields__):
            # create field
            field_value = field()

            object.__setattr__(field_value, "name", key)
            object.__setattr__(field_value, "type", type(value))

            # register it to class
            self.__undeclared_fields__.update({key: field_value})


class ImmutableInstanceError(Exception):
    pass


class _explicitSetter:
    """Register dataclass fields"""

    __immutable_treeclass__ = True

    def __setattr__(self, key, value):
        if self.__immutable_treeclass__:
            raise ImmutableInstanceError(
                f"Cannot set {key} = {value}. Use `.at['{key}'].set({value!r})` instead."
            )

        object.__setattr__(self, key, value)
