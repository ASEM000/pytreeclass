# This script defines the base class for the tree classes.
# the base class handles flatten/unflatten rules for the tree classes

from __future__ import annotations

from dataclasses import MISSING
from typing import Any

import pytreeclass as pytc


class _fieldDict(dict):
    """A dict used for `__pytree_structure__` attribute of a treeclass instance"""

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def _tree_structure(tree) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return dynamic and static fields of the pytree instance"""
    # this function classifies tree vars into trainable/untrainable dict items
    # and returns a tuple of two dicts (dynamic, static)
    # that mark the tree leaves seen by JAX computations and the static(tree structure) that are
    # not seen by JAX computations. the scanning is done if the instance is not frozen.
    # otherwise the cached values are returned.

    static, dynamic = _fieldDict(tree.__dict__), _fieldDict()

    for field_item in pytc.fields(tree).values():
        if not field_item.metadata.get("static", False):
            dynamic[field_item.name] = static.pop(field_item.name)

    return (dynamic, static)


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
