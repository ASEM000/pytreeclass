from __future__ import annotations

import functools as ft
import inspect
from dataclasses import dataclass

import jax

from pytreeclass.src.tree_base import implicitTreeBase, treeBase
from pytreeclass.src.tree_indexer import treeIndexer
from pytreeclass.src.tree_op_base import treeOpBase


class ImmutableInstanceError(Exception):
    pass


def treeclass(*args, **kwargs):
    """Class JAX  compaitable decorator for `dataclass`"""

    def class_wrapper(cls, field_only: bool):

        user_defined_init = "__init__" in cls.__dict__
        dCls = dataclass(init=not user_defined_init, repr=False, eq=False)(cls)

        base_classes = (dCls, treeBase)
        base_classes += (treeOpBase, treeIndexer)
        base_classes += (implicitTreeBase,) if not field_only else ()
        new_cls = type(cls.__name__, base_classes, {})

        def immutable_setattr(self, key, value):
            if self.__immutable_treeclass__:
                raise ImmutableInstanceError(
                    f"Cannot set {key} = {value}. Use `.at['{key}'].set({value!r})` instead."
                )
            # execute original setattr
            mutable_setattr(self, key, value)

        def immutate_post_method(func):
            @ft.wraps(func)
            def wrapper(self, *args, **kwargs):
                # modify instance mutable behavior
                object.__setattr__(self, "__immutable_treeclass__", False)
                # execute `func` with mutable behavior
                func(self, *args, **kwargs)
                # set instance as immutable after `func` is executed.
                object.__setattr__(self, "__immutable_treeclass__", True)

            return wrapper

        mutable_setattr = new_cls.__setattr__
        new_cls.__immutable_treeclass__ = True
        new_cls.__setattr__ = immutable_setattr
        new_cls.__init__ = immutate_post_method(new_cls.__init__)

        return jax.tree_util.register_pytree_node_class(new_cls)

    if len(args) == 1 and inspect.isclass(args[0]):
        return class_wrapper(args[0], field_only=False)

    elif len(args) == 0 and len(kwargs) > 0:
        field_only = kwargs.get("field_only", False)
        return ft.partial(class_wrapper, field_only=field_only)
