from __future__ import annotations

import dataclasses as dc
import functools as ft
import math
import operator as op
from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from pytreeclass._src.tree_decorator import (
    _FIELD_MAP,
    _FROZEN,
    _NOT_SET,
    _POST_INIT,
    Field,
    _generate_field_map,
    _get_init_method,
)
from pytreeclass._src.tree_freeze import unfreeze
from pytreeclass._src.tree_indexer import _at_indexer
from pytreeclass._src.tree_operator import _copy, _hash, bcmap
from pytreeclass._src.tree_pprint import tree_repr, tree_str

PyTree = Any


def _setattr(tree: PyTree, key: str, value: Any) -> None:
    """Set the attribute of the tree if the tree is not frozen"""
    if getattr(tree, _FROZEN):
        msg = f"Cannot set {key}={value!r}. Use `.at['{key}'].set({value!r})` instead."
        raise dc.FrozenInstanceError(msg)

    tree.__dict__[key] = value

    if hasattr(value, _FIELD_MAP) and (key not in getattr(tree, _FIELD_MAP)):
        field = Field(name=key, type=type(value))  # type: ignore
        # register it to dataclass fields
        getattr(tree, _FIELD_MAP)[key] = field


def _delattr(tree, key: str) -> None:
    """Delete the attribute of the  if tree is not frozen"""
    if getattr(tree, _FROZEN):
        raise dc.FrozenInstanceError(f"Cannot delete {key}.")
    del tree.__dict__[key]


def _new_wrapper(new_func):
    @ft.wraps(new_func)
    def new_method(cls, *_, **__) -> PyTree:
        self = object.__new__(cls)
        for field in cls.__dict__[_FIELD_MAP].values():
            if field.default is not _NOT_SET:
                self.__dict__[field.name] = field.default
            elif field.default_factory is not None:
                self.__dict__[field.name] = field.default_factory()
        return self

    return new_method


def _apply_callbacks(tree, init: bool = True):
    for field in tree.__class__.__dict__[_FIELD_MAP].values():
        # init means that we are validating fields that are initialized
        if field.init is not init or field.callbacks is None:
            continue

        for callback in field.callbacks:
            try:
                # callback is a function that takes the value of the field
                # and returns a modified value
                value = callback(getattr(tree, field.name))
                setattr(tree, field.name, value)
            except Exception as e:
                stage = "__init__" if init else "__post_init__"
                msg = f"Error at `{stage}` for field=`{field.name}`:\n{e}"
                raise type(e)(msg)


def _init_wrapper(init_func):
    @ft.wraps(init_func)
    def init_method(self, *a, **k) -> None:
        self.__dict__[_FROZEN] = False
        output = init_func(self, *a, **k)
        print("post init at", init_func.__name__)
        # call callbacks on fields that are initialized
        _apply_callbacks(self, init=True)

        # in case __post_init__ is defined then call it
        # after the tree is initialized
        # here, we assume that __post_init__ is a method
        if _POST_INIT in vars(self.__class__):
            output = getattr(self, _POST_INIT)()
            # call validation on fields that are not initialized
            _apply_callbacks(self, init=False)

        # handle freezing values and uninitialized fields
        for field in self.__class__.__dict__[_FIELD_MAP].values():
            if field.name not in vars(self):
                # at this point, all fields should be initialized
                # in principle, this error will be caught when invoking `repr`/`str`
                # like in `dataclasses` but we raise it here for better error message.
                raise AttributeError(f"field=`{field.name}` is not initialized.")

        # output must be None,otherwise will raise error
        self.__dict__[_FROZEN] = True
        return output

    return init_method


def _get_wrapper(get_func):
    @ft.wraps(get_func)
    def get_method(self, key: str) -> Any:
        # avoid non-scalar error, raised by `jax` transformation
        # if a frozen value is returned.
        try:
            return unfreeze(get_func(self, key))
        except AttributeError as error:
            raise type(error)(error)

    return get_method


def _flatten(tree) -> tuple[Any, tuple[str, dict[str, Any]]]:
    """Flatten rule for `jax.tree_flatten`"""
    # in essence anything not declared in dataclass fields will be considered static
    static, dynamic = dict(vars(tree)), dict()
    for key in vars(tree.__class__)[_FIELD_MAP]:
        dynamic[key] = static.pop(key)

    return dynamic.values(), (dynamic.keys(), static)


def _unflatten(cls, treedef, leaves):
    """Unflatten rule for `jax.tree_unflatten`"""
    tree = object.__new__(cls)  # do not call cls constructor
    tree.__dict__.update(treedef[1])
    tree.__dict__.update(zip(treedef[0], leaves))
    return tree


def _treeclass(cls, order):
    # check if the input is a valid class
    if not isinstance(cls, type):
        # non class input will raise an error
        msg = f"@treeclass accepts class as input. Found type={type(cls)}"
        raise TypeError(msg)

    no_methods = ("__setattr__", "__delattr__", "__getattribute__", "__getattr__")
    # check if the class has any of the above methods
    for method_name in no_methods:
        if method_name in vars(cls):
            msg = f"Cannot define `{method_name}` in {cls.__name__}."
            raise AttributeError(msg)

    # get the mapping between field names and field `NamedTuple` objects
    field_map = _generate_field_map(cls)
    # generate init method if not defined otherwise use the existing one
    init_method = _get_init_method(cls, field_map)

    # treeclass constants
    setattr(cls, _FIELD_MAP, field_map)
    setattr(cls, _FROZEN, False)
    # initialize the class
    setattr(cls, "__new__", _new_wrapper(cls.__new__))
    setattr(cls, "__init__", _init_wrapper(init_method))
    # immutable methods
    setattr(cls, "__getattribute__", _get_wrapper(cls.__getattribute__))
    setattr(cls, "__setattr__", _setattr)
    setattr(cls, "__delattr__", _delattr)
    # pretty printing
    setattr(cls, "__repr__", tree_repr)
    setattr(cls, "__str__", tree_str)
    # JAX tree utilities
    setattr(cls, "tree_flatten", _flatten)
    setattr(cls, "tree_unflatten", classmethod(_unflatten))
    # at indexing
    setattr(cls, "at", property(_at_indexer))
    # copy and hash
    setattr(cls, "__copy__", _copy)
    setattr(cls, "__hash__", _hash)

    if order is True:
        # math operations
        setattr(cls, "__abs__", bcmap(op.abs))
        setattr(cls, "__add__", bcmap(op.add))
        setattr(cls, "__and__", bcmap(op.and_))
        setattr(cls, "__ceil__", bcmap(math.ceil))
        setattr(cls, "__copy__", _copy)
        setattr(cls, "__divmod__", bcmap(divmod))
        setattr(cls, "__eq__", bcmap(op.eq))
        setattr(cls, "__floor__", bcmap(math.floor))
        setattr(cls, "__floordiv__", bcmap(op.floordiv))
        setattr(cls, "__ge__", bcmap(op.ge))
        setattr(cls, "__gt__", bcmap(op.gt))
        setattr(cls, "__inv__", bcmap(op.inv))
        setattr(cls, "__invert__", bcmap(op.invert))
        setattr(cls, "__le__", bcmap(op.le))
        setattr(cls, "__lshift__", bcmap(op.lshift))
        setattr(cls, "__lt__", bcmap(op.lt))
        setattr(cls, "__matmul__", bcmap(op.matmul))
        setattr(cls, "__mod__", bcmap(op.mod))
        setattr(cls, "__mul__", bcmap(op.mul))
        setattr(cls, "__ne__", bcmap(op.ne))
        setattr(cls, "__neg__", bcmap(op.neg))
        setattr(cls, "__or__", bcmap(op.or_))
        setattr(cls, "__pos__", bcmap(op.pos))
        setattr(cls, "__pow__", bcmap(op.pow))
        setattr(cls, "__radd__", bcmap(op.add))
        setattr(cls, "__rand__", bcmap(op.and_))
        setattr(cls, "__rdivmod__", bcmap(divmod))
        setattr(cls, "__rfloordiv__", bcmap(op.floordiv))
        setattr(cls, "__rlshift__", bcmap(op.lshift))
        setattr(cls, "__rmatmul__", bcmap(op.matmul))
        setattr(cls, "__rmod__", bcmap(op.mod))
        setattr(cls, "__rmul__", bcmap(op.mul))
        setattr(cls, "__ror__", bcmap(op.or_))
        setattr(cls, "__round__", bcmap(round))
        setattr(cls, "__rpow__", bcmap(op.pow))
        setattr(cls, "__rrshift__", bcmap(op.rshift))
        setattr(cls, "__rshift__", bcmap(op.rshift))
        setattr(cls, "__rsub__", bcmap(op.sub))
        setattr(cls, "__rtruediv__", bcmap(op.truediv))
        setattr(cls, "__rxor__", bcmap(op.xor))
        setattr(cls, "__sub__", bcmap(op.sub))
        setattr(cls, "__truediv__", bcmap(op.truediv))
        setattr(cls, "__xor__", bcmap(op.xor))

    return jtu.register_pytree_node_class(cls)


def treeclass(cls=None, *, order: bool = True):
    """Decorator to convert a class to a `treeclass`

    Example:
        >>> @treeclass
        ... class Tree:
        ...     x: float
        ...     y: float
        ...     z: float

        >>> tree = Tree(1, 2, 3)

        >>> tree
        Tree(x=1, y=2, z=3)

    Args:
        cls: class to be converted to a `treeclass`
        order: if `True` the `treeclass` math operations will be applied leaf-wise (default: `True`)

    Returns:`
        `treeclass` of the input class

    Raises:
        TypeError: if the input class is not a `class`
    """
    if cls is None:
        return ft.wraps(cls)(ft.partial(_treeclass, order=order))
    return _treeclass(cls, order)  # @treeclass(...)


def is_tree_equal(lhs: Any, rhs: Any) -> bool:
    """Assert if two pytrees are equal"""
    lhs_leaves, lhs_treedef = jtu.tree_flatten(lhs)
    rhs_leaves, rhs_treedef = jtu.tree_flatten(rhs)

    if not (lhs_treedef == rhs_treedef):
        return False

    for (lhs, rhs) in zip(lhs_leaves, rhs_leaves):
        if isinstance(lhs, (jnp.ndarray, np.ndarray)):
            if isinstance(rhs, (jnp.ndarray, np.ndarray)):
                if not np.array_equal(lhs, rhs):
                    return False
            else:
                return False
        else:
            return lhs == rhs
    return True
