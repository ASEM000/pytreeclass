from __future__ import annotations

import functools as ft
import inspect
from dataclasses import dataclass

import jax

from pytreeclass.src.misc import _mutable
from pytreeclass.src.tree_base import _explicitSetter, _implicitSetter, _treeBase
from pytreeclass.src.tree_indexer import _treeIndexer
from pytreeclass.src.tree_op_base import _treeOpBase


def treeclass(*args, **kwargs):
    def class_wrapper(cls, field_only: bool):

        dCls = dataclass(
            init="__init__" not in cls.__dict__,
            repr=False,
            eq=False,
            unsafe_hash=False,
            order=False,
            frozen=False,
        )(cls)

        base_classes = (dCls,)
        base_classes += (_explicitSetter,) if field_only else (_implicitSetter,)
        base_classes += (_treeIndexer, _treeOpBase)
        base_classes += (_treeBase,)

        new_cls = type(cls.__name__, base_classes, {})

        new_cls.__init__ = _mutable(new_cls.__init__)

        return jax.tree_util.register_pytree_node_class(new_cls)

    if len(args) == 1 and inspect.isclass(args[0]):
        # @treeclass
        return class_wrapper(args[0], field_only=False)

    elif len(args) == 0 and len(kwargs) > 0:
        # @treeclass(...)
        field_only = kwargs.get("field_only", False)
        return ft.partial(class_wrapper, field_only=field_only)
