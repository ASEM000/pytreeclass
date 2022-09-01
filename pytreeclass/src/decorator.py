from __future__ import annotations

import functools as ft
import inspect
from dataclasses import dataclass

import jax

from pytreeclass.src.misc import ImmutableInstanceError, _mutable
from pytreeclass.src.tree_base import _implicitTreeBase, _treeBase
from pytreeclass.src.tree_indexer import _treeIndexer
from pytreeclass.src.tree_op_base import _treeOpBase


def treeclass(*args, **kwargs):
    def immutable_setattr(mutable_setattr):
        # temporary disable mutable behavior
        # to allow for instance attribute setting
        # in __init__ method
        def wrapper(self, key, value):
            if self.__immutable_treeclass__:
                raise ImmutableInstanceError(
                    f"Cannot set {key} = {value}. Use `.at['{key}'].set({value!r})` instead."
                )
            # execute original setattr
            mutable_setattr(self, key, value)

        return wrapper

    def class_wrapper(cls, field_only: bool):

        dCls = dataclass(
            init="__init__" not in cls.__dict__,
            repr=False,
            eq=False,
            unsafe_hash=False,
            order=False,
        )(cls)

        base_classes = (_implicitTreeBase,) if not field_only else ()
        base_classes += (_treeBase,)
        base_classes += (_treeIndexer, _treeOpBase)
        base_classes += (dCls,)

        new_cls = type(cls.__name__, base_classes, {})

        mutable_setattr = new_cls.__setattr__
        new_cls.__immutable_treeclass__ = True
        new_cls.__setattr__ = immutable_setattr(mutable_setattr)
        new_cls.__init__ = _mutable(new_cls.__init__)

        return jax.tree_util.register_pytree_node_class(new_cls)

    if len(args) == 1 and inspect.isclass(args[0]):
        # @treeclass
        return class_wrapper(args[0], field_only=False)

    elif len(args) == 0 and len(kwargs) > 0:
        # @treeclass(...)
        field_only = kwargs.get("field_only", False)
        return ft.partial(class_wrapper, field_only=field_only)
