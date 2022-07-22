from __future__ import annotations

from dataclasses import MISSING, field
from typing import Any

import jax

from .decorator_util import cached_property
from .tree_util import is_treeclass, is_treeclass_leaf
from .tree_viz import tree_indent, tree_str


class treeBase:

    def __setattr__(self, name, value):
        if hasattr(self, "frozen_treeclass") :
            raise ValueError("Cannot set value to a frozen treeclass.")
        object.__setattr__(self, name, value)

    @cached_property
    def tree_fields(self):
        # freeze the treeclass once the tree is traversed.
        object.__setattr__(self, "frozen_treeclass", True)
        
        static, dynamic = dict(), dict()
        # register other variables defined in other context
        # if their values is an instance of treeclass
        # leaves seen by jax is frozen once tree_fields is called.
        # this enable the user to create leaves after the instantiation of the class
        # However , freezes it once a JAX operation that requires tree_flatten is applied
        # or in general tree_fields/tree_flatten is called
        # this design enables avoiding declaration repeatition in dataclass fields
        # tree_fields is called in tree_viz, repr, and str operations.
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

            # str is always excluded
            excluded_by_type = isinstance(value, str)
            excluded_by_meta = ("static" in fi.metadata) and fi.metadata["static"] is True  # fmt: skip

            if excluded_by_type or excluded_by_meta:
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

        newCls = cls.__new__(cls)
        for k, v in attrs.items():
            object.__setattr__(newCls, k, v)
        return newCls

    @cached_property
    def treeclass_leaves(self):
        return jax.tree_util.tree_leaves(self, is_treeclass_leaf)

    @cached_property
    def flatten_leaves(self):
        return jax.tree_util.tree_flatten(self)

    def __hash__(self):
        return hash(tuple(*self.flatten_leaves))

    @cached_property
    def __treeclass_repr__(self):
        return tree_indent(self)

    @cached_property
    def __treeclass_str__(self):
        return tree_str(self)

    def __repr__(self):
        return self.__treeclass_repr__

    def __str__(self):
        return self.__treeclass_str__

    def asdict(self):
        return {**self.tree_fields[0], **self.tree_fields[1]}

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
