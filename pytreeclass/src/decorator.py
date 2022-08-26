from __future__ import annotations

import functools as ft
import inspect
from dataclasses import dataclass

import jax

from pytreeclass.src.misc import ImmutableInstanceError, mutableContext
from pytreeclass.src.tree_base import _implicitSetter, _treeBase
from pytreeclass.src.tree_indexer import _treeIndexer
from pytreeclass.src.tree_op_base import _treeOpBase


def treeclass(*args, **kwargs):
    """Class JAX  compaitable decorator for `dataclass`"""

    def class_wrapper(cls, field_only: bool):

        user_defined_init = "__init__" in cls.__dict__
        dCls = dataclass(init=not user_defined_init, repr=False, eq=False)(cls)

        base_classes = (dCls, _treeBase, _treeOpBase, _treeIndexer)
        base_classes += (_implicitSetter,) if not field_only else ()
        new_cls = type(cls.__name__, base_classes, {})

        def immutable_setattr(mutable_setattr):
            def wrapper(self, key, value):
                if self.__immutable_treeclass__:
                    raise ImmutableInstanceError(
                        f"Cannot set {key} = {value}. Use `.at['{key}'].set({value!r})` instead."
                    )
                # execute original setattr
                mutable_setattr(self, key, value)

            return wrapper

        def immutate_post_method(func):
            @ft.wraps(func)
            def wrapper(self, *args, **kwargs):
                # execute `func` with mutable behavior
                with mutableContext(self):
                    func(self, *args, **kwargs)

            return wrapper

        mutable_setattr = new_cls.__setattr__
        new_cls.__immutable_treeclass__ = True
        new_cls.__setattr__ = immutable_setattr(mutable_setattr)
        new_cls.__init__ = immutate_post_method(new_cls.__init__)

        return jax.tree_util.register_pytree_node_class(new_cls)

    if len(args) == 1 and inspect.isclass(args[0]):
        return class_wrapper(args[0], field_only=False)

    elif len(args) == 0 and len(kwargs) > 0:
        field_only = kwargs.get("field_only", False)
        return ft.partial(class_wrapper, field_only=field_only)
