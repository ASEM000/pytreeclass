from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass

import jax

from pytreeclass.src.tree_base import explicitTreeBase, implicitTreeBase, treeBase
from pytreeclass.src.tree_indexer import treeIndexer
from pytreeclass.src.tree_op_base import treeOpBase

class ImmutablInstanceError(Exception):
    pass

def _immutate_treeclass(cls):

    cls.__immutable_treeclass__ = False
    mutable_setattr = cls.__setattr__

    def immutable_setattr(self, key, value):
        if self.__immutable_treeclass__:
            raise ImmutablInstanceError(f"Cannot set {key} = {value}.")
        mutable_setattr(self, key, value)

    def execute_post_init(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)

            # post inititialization
            object.__setattr__(self, "__immutable_treeclass__", True)

        return wrapper

    cls.__setattr__ = immutable_setattr
    cls.__init__ = execute_post_init(cls.__init__)

    return cls


def treeclass(*args, **kwargs):
    """Class JAX  compaitable decorator for `dataclass`"""

    def wrapper(cls, field_only: bool):
        user_defined_init = "__init__" in cls.__dict__

        dCls = dataclass(
            unsafe_hash=True, init=not user_defined_init, repr=False, eq=False
        )(cls)

        base_classes = (dCls, treeBase)

        base_classes += (treeOpBase, treeIndexer)
        base_classes += (explicitTreeBase,) if field_only else (implicitTreeBase,)

        new_cls = _immutate_treeclass(type(cls.__name__, base_classes, {}))

        return jax.tree_util.register_pytree_node_class(new_cls)

    if len(args) == 1 and inspect.isclass(args[0]):
        return wrapper(args[0], field_only=False)

    elif len(args) == 0 and len(kwargs) > 0:
        field_only = kwargs.get("field_only", False)
        return functools.partial(wrapper, field_only=field_only)
