from __future__ import annotations

from dataclasses import MISSING, field
from typing import Any, Sequence

import jax.numpy as jnp
import jax.tree_util as jtu

from pytreeclass.src.tree_util import (
    _freeze_nodes,
    _unfreeze_nodes,
    is_treeclass,
    is_treeclass_leaf,
    static_value,
)
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
    def freeze(self) -> PyTree:
        """Freeze treeclass.

        Returns:
            New frozen instance.

        Example:

        >>> model = model.freeze()
        >>> assert model.frozen == True
        """
        return _freeze_nodes(jtu.tree_unflatten(*jtu.tree_flatten(self)[::-1]))

    def unfreeze(self) -> PyTree:
        """Unfreeze treeclass.

        Returns:
            New unfrozen instance.

        Example :

        >>> model = model.unfreeze()
        >>> assert model.frozen == False
        """
        return _unfreeze_nodes(jtu.tree_unflatten(*jtu.tree_flatten(self)[::-1]))

    @property
    def frozen(self) -> bool:
        """Show treeclass frozen status.

        Returns:
            Frozen state boolean.
        """
        return True if hasattr(self, "__frozen_tree_fields__") else False

    def __setattr__(self, name: str, value: Any) -> None:
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

        if self.frozen:
            return (), ((), {"__frozen_tree_fields__": (dynamic, static)})

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
        new_cls = cls.__new__(cls)

        tree_fields = treedef[1].get("__frozen_tree_fields__", None)

        if tree_fields is not None:
            object.__setattr__(new_cls, "__frozen_tree_fields__", tree_fields)
            dynamic, static = tree_fields
            attrs = {**dynamic, **static}

        else:

            dynamic_vals, dynamic_keys = children, treedef[0]
            static_keys, static_vals = treedef[1].keys(), treedef[1].values()

            attrs = dict(
                zip(
                    (*dynamic_keys, *static_keys),
                    (*dynamic_vals, *static_vals),
                )
            )

        for k, v in attrs.items():
            object.__setattr__(new_cls, k, v)
        return new_cls

    @property
    def treeclass_leaves(self) -> Sequence[PyTree | Any, ...]:
        """Tree leaves of treeclass

        Example:

            @pytc.treeclass
            class T0:
                a : int = 1
                b : int = 2

            @pytc.treeclass
            class T1:
                c : T0 = T0()
                d : int = 3


            @pytc.treeclass
            class T2 :
                e : T1 = T1()
                f : T0 = T0()
                g : int = 4

            >>> t = T2()

            >>> print(t.tree_diagram())
            T2
                ├── e=T1
                │   ├── c=T0
                │   │   ├── a=1
                │   │   └── b=2
                │   └── d=3
                ├── f=T0
                │   ├── a=1
                │   └── b=2
                └── g=4

            >>> print(t.treeclass_leaves)
            [T0(a=1,b=2), 3, T0(a=1,b=2), 4]
        """
        return jtu.tree_leaves(self, is_treeclass_leaf)

    def __hash__(self):
        return hash(tuple(*jtu.tree_flatten(self)))

    def asdict(self) -> dict[str, Any]:
        """Dictionary representation of dataclass_fields"""
        dynamic, static = self.__tree_fields__
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

        if name not in self.__dataclass_fields__:
            # create field
            field_value = field(repr=repr, metadata={"static": static})

            object.__setattr__(field_value, "name", name)
            object.__setattr__(field_value, "type", type(node))

            # register it to class
            self.__dataclass_fields__.update({name: field_value})
            object.__setattr__(self, name, node)

        return self.__dict__[name]

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

    def __generate_tree_fields__(self) -> tuple[dict[str, Any], dict[str, Any]]:
        dynamic, static = fieldDict(), fieldDict()
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

            excluded_by_meta = fi.metadata.get("static", False)
            excluded_by_type = isinstance(value, static_value)

            if excluded_by_type or excluded_by_meta:
                static[fi.name] = value

            else:
                dynamic[fi.name] = value

        return (dynamic, static)


class explicitTreeBase:
    """ "Register  dataclass fields only"""

    @property
    def __tree_fields__(self):
        """Computes the dynamic and static fields.

        Returns:
            Pair of dynamic and static dictionaries.
        """
        if self.__dict__.get("__frozen_tree_fields__", None) is not None:
            return self.__frozen_tree_fields__
        else:
            return self.__generate_tree_fields__()


class implicitTreeBase:
    """Register dataclass fields and treeclass instance variables"""

    def __register_treeclass_instance_variables__(self) -> None:
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

    @property
    def __tree_fields__(self):
        """Computes the dynamic and static fields.

        Returns:
            Pair of dynamic and static dictionaries.
        """
        if self.__dict__.get("__frozen_tree_fields__", None) is not None:
            return self.__frozen_tree_fields__

        else:
            self.__register_treeclass_instance_variables__()
            return self.__generate_tree_fields__()
