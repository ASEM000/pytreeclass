from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass

import jax

from pytreeclass.src.decorator_util import _immutate_treeclass
from pytreeclass.src.tree_base import explicitTreeBase, implicitTreeBase, treeBase
from pytreeclass.src.tree_indexer import treeIndexer
from pytreeclass.src.tree_op_base import treeOpBase


def treeclass(*args, **kwargs):
    """Class JAX  compaitable decorator for `dataclass`"""

    def wrapper(cls, field_only: bool, immutable: bool):
        user_defined_init = "__init__" in cls.__dict__

        dCls = dataclass(
            unsafe_hash=True, init=not user_defined_init, repr=False, eq=False
        )(cls)

        base_classes = (dCls, treeBase)

        base_classes += (treeOpBase, treeIndexer)
        base_classes += (explicitTreeBase,) if field_only else (implicitTreeBase,)

        new_cls = type(cls.__name__, base_classes, {})
        new_cls = _immutate_treeclass(new_cls) if immutable else new_cls

        return jax.tree_util.register_pytree_node_class(new_cls)

    if len(args) == 1 and inspect.isclass(args[0]):
        return wrapper(args[0], field_only=False, immutable=True)

    elif len(args) == 0 and len(kwargs) > 0:
        field_only = kwargs.get("field_only", False)
        immutable = kwargs.get("immutable", True)
        return functools.partial(wrapper, field_only=field_only, immutable=immutable)
