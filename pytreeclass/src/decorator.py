from __future__ import annotations

import functools
import inspect
import warnings
from dataclasses import dataclass, field

import jax

from .tree_base import treeBase, treeOpBase


def static_field(**kwargs):
    """ignore from pytree computations"""
    return field(**{**kwargs, **{"metadata": {"static": True}}})


def treeclass(*args, **kwargs):
    """Class JAX  compaitable decorator for `dataclass`"""

    def wrapper(cls, op: bool = True):
        user_defined_init = "__init__" in cls.__dict__

        dCls = dataclass(unsafe_hash=True,
                         init=not user_defined_init,
                         repr=False,
                         eq=False)(cls)

        base_classes = (dCls, treeBase)
        base_classes += (treeOpBase, ) if op else ()

        newCls = type(cls.__name__, base_classes, {})

        return jax.tree_util.register_pytree_node_class(newCls)

    if len(args) > 0 and inspect.isclass(args[0]):
        return wrapper(args[0], True)

    elif len(args) == 0 and len(kwargs) > 0:
        op = kwargs["op"] if "op" in kwargs else False
        return functools.partial(wrapper, op=op)


def run_once(*args, **kwargs):

    def run_once_wrapper(func, raise_error=True):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not wrapper.run:
                result = func(*args, **kwargs)
                wrapper.run = True
                return result
            else:
                if raise_error:
                    raise ValueError(
                        f"Function {func!r} is wrapped with @run_once is called more than once."
                    )
                else:
                    warnings.warn(
                        f"Function {func!r} wrapped with @run_once is called more than once."
                    )

        wrapper.run = False
        return wrapper

    if len(args) > 0 and inspect.isfunction(args[0]):
        return run_once_wrapper(args[0], True)

    elif len(args) == 0 and len(kwargs) > 0:
        raise_error = kwargs[
            "raise_error"] if "raise_error" in kwargs else False
        return functools.partial(run_once_wrapper, raise_error=raise_error)
