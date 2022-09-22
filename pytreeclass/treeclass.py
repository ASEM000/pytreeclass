from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import jax

from pytreeclass._src.misc import _mutable
from pytreeclass._src.tree_base import _treeBase
from pytreeclass._src.tree_indexer import _treeIndexer
from pytreeclass._src.tree_op import _treeOp
from pytreeclass._src.tree_pretty import _treePretty
from pytreeclass._src.tree_util import _tree_fields


class ImmutableInstanceError(Exception):
    pass


def treeclass(*args, **kwargs):
    def immutable_setter(tree, key: str, value: Any) -> None:

        if tree.__immutable_pytree__:
            raise ImmutableInstanceError(f"Cannot set {key}. Use `.at['{key}'].set({value!r})` instead.")  # fmt: skip

        object.__setattr__(tree, key, value)

        if (isinstance(value, _treeBase)) and (key not in _tree_fields(tree)):
            # create field
            field_value = field()

            object.__setattr__(field_value, "name", key)
            object.__setattr__(field_value, "type", type(value))

            # register it to class
            new_fields = {**tree.__undeclared_fields__, **{key: field_value}}  # fmt: skip
            object.__setattr__(tree, "__undeclared_fields__", MappingProxyType(new_fields))  # fmt: skip

    def class_wrapper(cls):

        if "__setattr__" in vars(cls):
            raise AttributeError("`treeclass` cannot be applied to class with `__setattr__` method.")  # fmt: skip

        dCls = dataclass(
            init="__init__" not in vars(cls),
            repr=False,  # repr is handled by _treePretty
            eq=False,  # eq is handled by _treeOpBase
            unsafe_hash=False,  # hash is handled by _treeOpBase
            order=False,  # order is handled by _treeOpBase
            frozen=False,  # frozen is `immutable_setter`
        )(cls)

        new_cls = type(
            cls.__name__,
            (dCls, _treeIndexer, _treeOp, _treePretty, _treeBase),
            {
                "__immutable_pytree__": True,
                "__undeclared_fields__": MappingProxyType({}),
                "__setattr__": immutable_setter,
            },
        )

        # temporarily mutate the tree instance to execute the __init__ method
        # without raising `__immutable_treeclass__` error
        # then restore the tree original immutable behavior after the function is called
        # _mutable can be applied to any class method that is decorated with @treeclass
        # to temporarily make the class mutable
        # however, it is not recommended to use it outside of __init__ method
        new_cls.__init__ = _mutable(new_cls.__init__)

        return jax.tree_util.register_pytree_node_class(new_cls)

    if len(args) == 1 and inspect.isclass(args[0]):
        # no args are passed to the decorator (i.e. @treeclass)
        return class_wrapper(args[0])
    else:
        raise TypeError(
            f"Input argument to `treeclass` must be of `class` type. Found {(*args, *kwargs)}."
        )
