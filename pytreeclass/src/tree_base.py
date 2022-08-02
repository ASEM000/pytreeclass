from __future__ import annotations

import copy
from dataclasses import MISSING, field
from typing import Any

from jax.tree_util import tree_leaves

from .tree_util import _freeze_nodes, _unfreeze_nodes, is_treeclass, is_treeclass_leaf
from .tree_viz import summary, tree_box, tree_diagram, tree_indent, tree_str


def tree_fields(self):
    static, dynamic = dict(), dict()
    # register other variables defined in other context
    # if their value is an instance of treeclass
    # to avoid redefining them as dataclass fields.

    # register all dataclass fields
    for fi in self.__dataclass_fields__.values():
        # field value is defined in class dict
        if fi.name in self.__dict__:
            value = self.__dict__[fi.name]

        # field value is defined in field default
        elif fi.default is not MISSING:
            self.__dict__[fi.name] = fi.default
            value = fi.default

        else:
            # the user did not declare a variable defined in field
            raise ValueError(f"field={fi.name} is not declared.")

        # if the parent is frozen, freeze all dataclass fields children
        if self.frozen:
            static[fi.name] = value

        else:
            # exclude any string
            # and mutate the class field static metadata for this variable for future instances
            excluded_by_type = isinstance(value, str)
            excluded_by_meta = ("static" in fi.metadata) and fi.metadata["static"] is True  # fmt: skip

            if excluded_by_type:
                # add static type to metadata to class and its instance
                static[fi.name] = value
                updated_field = self.__dataclass_fields__[fi.name]
                object.__setattr__(
                    updated_field,
                    "metadata",
                    {**updated_field.metadata, **{"static": True}},
                )
                self.__dataclass_fields__[fi.name] = updated_field

            elif excluded_by_meta:
                static[fi.name] = value

            else:
                dynamic[fi.name] = value

    return (dynamic, static)


def register_treeclass_instance_variables(self):
    for var_name, var_value in self.__dict__.items():
        # check if a variable in self.__dict__ is treeclass
        # that is not defined in fields
        if (
            isinstance(var_name, str)
            and is_treeclass(var_value)
            and var_name not in self.__dataclass_fields__
        ):

            # create field
            field_value = field()
            setattr(field_value, "name", var_name)
            setattr(field_value, "type", type(var_value))

            # register it to class
            self.__dataclass_fields__.update({var_name: field_value})


class treeBase:
    def freeze(self):
        """Freeze treeclass.

        Returns:
            New frozen instance.

        Example:

        >>> model = model.freeze()
        >>> assert model.frozen == True
        """
        return _freeze_nodes(copy.copy(self))

    def unfreeze(self):
        """Unfreeze treeclass.

        Returns:
            New unfrozen instance.

        Example :

        >>> model = model.unfreeze()
        >>> assert model.frozen == False
        """
        return _unfreeze_nodes(copy.copy(self))

    @property
    def frozen(self) -> bool:
        """Show treeclass frozen status.

        Returns:
            Frozen state boolean.
        """
        if hasattr(self, "__frozen_treeclass__"):
            return self.__frozen_treeclass__
        return False

    def __setattr__(self, name, value):
        if self.frozen:
            raise ValueError("Cannot set a value to a frozen treeclass.")
        object.__setattr__(self, name, value)

    def tree_flatten(self):
        """Flatten rule for `jax.tree_flatten`

        Returns:
            Tuple of dynamic values and (dynamic keys,static dict)
        """
        # we need to transfer the state for the next instance through static
        # we also need to retrieve it for the current instance

        dynamic, static = self.__tree_fields__

        cache = {"__frozen_treeclass__": self.frozen}

        if hasattr(self, "__frozen_tree_fields__"):
            cache = {**cache, **{"__frozen_tree_fields__": self.__frozen_tree_fields__}}

        return (dynamic.values(), (dynamic.keys(), static, cache))

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
        dynamic_vals, dynamic_keys = children, treedef[0]
        static_keys, static_vals = treedef[1].keys(), treedef[1].values()
        cache_keys, cache_vals = treedef[2].keys(), treedef[2].values()

        attrs = dict(
            zip(
                (*dynamic_keys, *static_keys, *cache_keys),
                (*dynamic_vals, *static_vals, *cache_vals),
            )
        )

        new_cls = cls.__new__(cls)
        for k, v in attrs.items():
            object.__setattr__(new_cls, k, v)
        return new_cls

    @property
    def treeclass_leaves(self):
        """Tree leaves of treeclass"""
        return tree_leaves(self, is_treeclass_leaf)

    def __hash__(self):
        return hash(tuple(*self.flatten_leaves))

    def asdict(self):
        """Dictionary representation of dataclass_fields"""
        dynamic, static = self.__tree_fields__
        return {**dynamic, **static}

    def register_node(
        self, node: Any, *, name: str, static: bool = False, repr: bool = True
    ) -> Any:
        """Add item to dataclass fields to bee seen by jax computations"""

        if name not in self.__dataclass_fields__:
            # create field
            field_value = field(repr=repr, metadata={"static": static})

            setattr(field_value, "name", name)
            setattr(field_value, "type", type(node))

            # register it to class
            self.__dataclass_fields__.update({name: field_value})
            self.__dict__[name] = node

        return self.__dict__[name]

    def __repr__(self):
        return tree_indent(self)

    def __str__(self):
        return tree_str(self)

    def summary(self, array=None):
        return summary(self, array)

    def tree_diagram(self):
        return tree_diagram(self)

    def tree_box(self, array=None):
        return tree_box(self, array)

    def __setitem__(self, key, value):
        """Set item in treeclass

        Args:
            key (_type_): _description_
            value (_type_): _description_

        Raises:
            ValueError: If treeclass is frozen raises error
        """
        if self.frozen:
            raise ValueError("Cannot set a value to a frozen treeclass.")
        self.__dict__[key] = value


class explicitTreeBase:
    """ "Register  dataclass fields only"""

    @property
    def __tree_fields__(self):
        """Computes the dynamic and static fields.

        Returns:
            Pair of dynamic and static dictionaries.
        """
        if self.frozen:
            if not hasattr(self, "__frozen_tree_fields__"):
                object.__setattr__(self, "__frozen_tree_fields__", tree_fields(self))

            return self.__frozen_tree_fields__

        return tree_fields(self)


class implicitTreeBase:
    """Register dataclass fields and treeclass instance variables"""

    @property
    def __tree_fields__(self):
        """Computes the dynamic and static fields.

        Returns:
            Pair of dynamic and static dictionaries.
        """
        if self.frozen:
            if not hasattr(self, "__frozen_tree_fields__"):
                register_treeclass_instance_variables(self)
                object.__setattr__(self, "__frozen_tree_fields__", tree_fields(self))

            return self.__frozen_tree_fields__

        else:
            register_treeclass_instance_variables(self)
            return tree_fields(self)
