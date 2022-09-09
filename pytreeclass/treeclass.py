from __future__ import annotations

import functools as ft
import inspect
from dataclasses import dataclass

import jax

from pytreeclass._src.misc import _mutable
from pytreeclass._src.tree_base import _explicitSetter, _implicitSetter, _treeBase
from pytreeclass._src.tree_indexer import _treeIndexer
from pytreeclass._src.tree_op import _treeOp
from pytreeclass._src.tree_pretty import _treePretty


def treeclass(*args, **kwargs):
    def class_wrapper(cls, field_only: bool):

        if "__setattr__" in cls.__dict__:
            raise AttributeError(
                "`treeclass` cannot be applied to class with `__setattr__` method."
            )

        dCls = dataclass(
            init="__init__" not in cls.__dict__,
            repr=False,  # repr is handled by _treePretty
            eq=False,  # eq is handled by _treeOpBase
            unsafe_hash=False,  # hash is handled by _treeOpBase
            order=False,  # order is handled by _treeOpBase
            frozen=False,  # frozen is handled by _explicitSetter/_implicitSetter
        )(cls)

        base_classes = (dCls,)
        base_classes += (_explicitSetter,) if field_only else (_implicitSetter,)
        base_classes += (_treeIndexer, _treeOp, _treePretty)
        base_classes += (_treeBase,)

        new_cls = type(cls.__name__, base_classes, {})
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
        return class_wrapper(args[0], field_only=False)

    elif len(args) == 0 and len(kwargs) > 0:
        # args are passed to the decorator (i.e. @treeclass(field_only=True))
        field_only = kwargs.get("field_only", False)
        return ft.partial(class_wrapper, field_only=field_only)
