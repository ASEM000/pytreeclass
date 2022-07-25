from __future__ import annotations

import copy
from dataclasses import MISSING, field
from typing import Any

import jax

from .tree_util import freeze_nodes, is_treeclass, is_treeclass_leaf, unfreeze_nodes
from .tree_viz import tree_indent, tree_str


class treeBase:
    def freeze(self):
        new_cls = copy.copy(self)
        return freeze_nodes(new_cls)

    def unfreeze(self):
        new_cls = copy.copy(self)
        return unfreeze_nodes(new_cls)

    @property
    def frozen(self):
        if hasattr(self, "__frozen_treeclass__"):
            return self.__frozen_treeclass__
        return False

    def __setattr__(self, name, value):
        if self.frozen:
            raise ValueError("Cannot set a value to a frozen treeclass.")
        object.__setattr__(self, name, value)

    @property
    def tree_fields(self):

        static, dynamic = dict(), dict()
        # register other variables defined in other context
        # if their value is an instance of treeclass
        # to avoid redefining them as dataclass fields.
        static["__frozen_treeclass__"] = self.frozen

        for var_name, var_value in self.__dict__.items():
            # check if a variable in self.__dict__ is treeclass
            # that is not defined in fields
            if (
                isinstance(var_name, str)
                and is_treeclass(var_value)
                and var_name not in self.__dataclass_fields__.keys()
            ):

                # create field
                field_value = field()
                setattr(field_value, "name", var_name)
                setattr(field_value, "type", type(var_value))

                # register it to class
                self.__dataclass_fields__.update({var_name: field_value})

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

            if self.frozen:
                static[fi.name] = value

            else:
                # str is always excluded
                excluded_by_type = isinstance(value, str)
                excluded_by_meta = ("static" in fi.metadata) and fi.metadata["static"] is True  # fmt: skip

                if excluded_by_type:
                    # add static type to metadata
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

    def tree_flatten(self):
        dynamic, static = self.tree_fields
        return (dynamic.values(), (dynamic.keys(), static))

    @classmethod
    def tree_unflatten(cls, aux, children):
        dynamic_vals, dynamic_keys = children, aux[0]

        static_keys, static_vals = aux[1].keys(), aux[1].values()
        attrs = dict(zip((*dynamic_keys, *static_keys), (*dynamic_vals, *static_vals)))

        new_cls = cls.__new__(cls)
        for k, v in attrs.items():
            object.__setattr__(new_cls, k, v)
        return new_cls

    @property
    def treeclass_leaves(self):
        return jax.tree_util.tree_leaves(self, is_treeclass_leaf)

    @property
    def flatten_leaves(self):
        return jax.tree_util.tree_flatten(self)

    def __hash__(self):
        return hash(tuple(*self.flatten_leaves))

    def __repr__(self):
        return tree_indent(self)

    def __str__(self):
        return tree_str(self)

    def asdict(self):
        dynamic, static = self.tree_fields
        static.pop("__frozen_treeclass__", None)
        return {**dynamic, **static}

    def register_node(self, node_defs: dict):
        def register_single_node(
            value: Any, key: str = None, static: bool = False, repr: bool = True
        ) -> Any:
            """add item to dataclass fields to bee seen by jax computations"""

            unnamed_count = sum([1 for k in self.__dict__ if k.startswith("unnamed")])
            field_key = f"unnamed_{unnamed_count}" if key is None else key

            # create field
            field_value = field(repr=repr, metadata={"static": static})

            setattr(field_value, "name", field_key)
            setattr(field_value, "type", type(value))

            # register it to class
            self.__dataclass_fields__.update({field_key: field_value})
            self.__dict__[field_key] = value

        for key, value in node_defs.items():
            register_single_node(key=key, value=value)
