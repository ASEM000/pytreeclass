# This script defines the base class for the tree classes.
# the base class handles flatten/unflatten rules for the tree classes

from __future__ import annotations

from dataclasses import MISSING

from pytreeclass._src.tree_util import _tree_structure


class _treeBase:
    def __new__(cls, *args, **kwargs):
        self = object.__new__(cls)
        for field_item in self.__dataclass_fields__.values():
            if field_item.default is not MISSING:
                object.__setattr__(self, field_item.name, field_item.default)
        return self

    def tree_flatten(self):
        """Flatten rule for `jax.tree_flatten`

        Returns:
            Tuple of dynamic values and (dynamic keys,static dict)
        """
        dynamic, static = _tree_structure(self)
        return dynamic.values(), (dynamic.keys(), static)

    @classmethod
    def tree_unflatten(cls, treedef, leaves):
        """Unflatten rule for `jax.tree_unflatten`

        Args:
            treedef:
                Pytree definition
                includes Dynamic nodes keys , static dictionary
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

        # update the instance values with the retrieved dynamic and static values
        self.__dict__.update(dict(zip(treedef[0], leaves)))
        self.__dict__.update(treedef[1])

        return self
