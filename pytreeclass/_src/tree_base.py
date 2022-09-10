# This script defines the base class for the tree classes.
# the base class handles flatten/unflatten rules for the tree classes

from __future__ import annotations

from dataclasses import MISSING, field
from typing import Any

from pytreeclass._src.tree_util import _tree_fields, _tree_structure


class _treeBase:
    def __new__(cls, *args, **kwargs):
        self = object.__new__(cls)
        object.__setattr__(self, "__undeclared_fields__", {})
        # set default values to class instance
        # Note: ideally this method should be called once
        for field_item in self.__dataclass_fields__.values():
            if field_item.default is not MISSING:
                object.__setattr__(self, field_item.name, field_item.default)

        return self

    def tree_flatten(self):
        """Flatten rule for `jax.tree_flatten`

        Returns:
            Tuple of dynamic values and (dynamic keys,static dict, cached values)
        """
        if hasattr(self, "__pytree_structure_cache__"):
            # return the cached pytree_structure
            return (), ((), (), (self.__pytree_structure_cache__))

        else:
            dynamic, static = _tree_structure(self)
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

        if len(treedef[2]) == 2:
            # retrieve the cached structure
            dynamic, static = treedef[2]
            # pass the cached structure to the new instance
            object.__setattr__(self, "__pytree_structure_cache__", (dynamic, static))
        else:
            dynamic = dict(zip(treedef[0], leaves))
            static = treedef[1]

        # update the instance values with the retrieved dynamic and static values
        self.__dict__.update(dynamic)
        self.__dict__.update(static)

        return self


class ImmutableInstanceError(Exception):
    pass


class _implicitSetter:
    """Register dataclass fields and treeclass instance variables"""

    __immutable_pytree__ = True

    def __setattr__(self, key: str, value: Any) -> None:

        if self.__immutable_pytree__:
            raise ImmutableInstanceError(
                f"Cannot set {key} = {value}. Use `.at['{key}'].set({value!r})` instead."
            )

        object.__setattr__(self, key, value)

        if (isinstance(value, _treeBase)) and (key not in _tree_fields(self)):
            # create field
            field_value = field()

            object.__setattr__(field_value, "name", key)
            object.__setattr__(field_value, "type", type(value))

            # register it to class
            self.__undeclared_fields__.update({key: field_value})


class _explicitSetter:
    """Register dataclass fields"""

    __immutable_pytree__ = True

    def __setattr__(self, key, value):
        if self.__immutable_pytree__:
            raise ImmutableInstanceError(
                f"Cannot set {key} = {value}. Use `.at['{key}'].set({value!r})` instead."
            )

        object.__setattr__(self, key, value)
