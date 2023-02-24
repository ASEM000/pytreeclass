from __future__ import annotations

import dataclasses as dc
import functools as ft
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
    _patch_init_method,
)
from pytreeclass._src.tree_freeze import unfreeze
from pytreeclass._src.tree_indexer import _TreeAtIndexer
from pytreeclass._src.tree_operator import _TreeOperator
from pytreeclass._src.tree_pprint import _TreePretty

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
        self.__dict__[_FROZEN] = False
        for field in cls.__dict__[_FIELD_MAP].values():
            if field.default is not _NOT_SET:
                setattr(self, field.name, field.default)
            elif field.default_factory is not None:
                setattr(self, field.name, field.default_factory())
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
        output = init_func(self, *a, **k)
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

    def decorator(cls, order):
        # check if the input is a valid class
        if not isinstance(cls, type):
            # non class input will raise an error
            msg = f"@treeclass accepts class as input. Found type={type(cls)}"
            raise TypeError(msg)

        for method_name in (
            "__setattr__",  # immutable setters
            "__delattr__",  # immutable deleters
            "__getattribute__",  # control freezable attributes access
            "__getattr__",  # can not override getattr
        ):
            if method_name in vars(cls):
                msg = f"Cannot define `{method_name}` in {cls.__name__}."
                raise AttributeError(msg)

        # generate and register field map to class.
        # generate init method if not defined based of the fields map
        # this is similar to dataclass decorator but only implements the relevant parts
        # for the purpose of this decorator.
        cls = _patch_init_method(cls)

        attrs = dict(vars(cls))

        # initialize class
        attrs.update(__new__=_new_wrapper(cls.__new__))
        attrs.update(__init__=_init_wrapper(cls.__init__))
        attrs.update(__getattribute__=_get_wrapper(cls.__getattribute__))

        # immutable setters/deleters
        attrs.update(__setattr__=_setattr)
        attrs.update(__delattr__=_delattr)

        # JAX flatten/unflatten rules
        attrs.update(tree_flatten=_flatten)
        attrs.update(tree_unflatten=classmethod(_unflatten))

        bases = (cls,)
        bases += (_TreeOperator,) if order else ()
        bases += (_TreePretty,) if repr else ()
        bases += (_TreeAtIndexer,)

        cls = type(cls.__name__, bases, attrs)

        # register the class to JAX registry
        return jtu.register_pytree_node_class(cls)

    if cls is None:
        return lambda cls: decorator(cls, order)  # @treeclass
    return decorator(cls, order)  # @treeclass(...)


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
