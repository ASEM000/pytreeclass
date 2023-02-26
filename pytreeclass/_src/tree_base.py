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
    _apply_callbacks,
    _generate_field_map,
    _retrieve_init_method,
)
from pytreeclass._src.tree_freeze import unfreeze
from pytreeclass._src.tree_indexer import _TreeIndexer, _TreeOperator
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
        for field in getattr(cls, _FIELD_MAP).values():
            if field.default is not _NOT_SET:
                self.__dict__[field.name] = field.default
            elif field.default_factory is not None:
                self.__dict__[field.name] = field.default_factory()
        return self

    return new_method


def _init_wrapper(init_func):
    @ft.wraps(init_func)
    def init_method(self, *a, **k) -> None:
        self.__dict__[_FROZEN] = False
        output = init_func(self, *a, **k)

        # call callbacks on fields that are initialized
        self.__dict__[_FROZEN] = False
        _apply_callbacks(self, init=True)

        # in case __post_init__ is defined then call it
        # after the tree is initialized
        # here, we assume that __post_init__ is a method
        if hasattr(self, _POST_INIT):
            # in case we found post_init in super class
            # then defreeze it first and call it
            # this behavior is differet to `dataclasses` with `frozen=True`
            # but similar if `frozen=False`
            # self.__dict__[_FROZEN] = False
            output = getattr(self, _POST_INIT)()
            # call validation on fields that are not initialized
            self.__dict__[_FROZEN] = False
            _apply_callbacks(self, init=False)

        # handle freezing values and uninitialized fields
        for field in getattr(self, _FIELD_MAP).values():
            if field.name not in vars(self):
                # at this point, all fields should be initialized
                # in principle, this error will be caught when invoking `repr`/`str`
                # like in `dataclasses` but we raise it here for better error message.
                raise AttributeError(f"field=`{field.name}` is not initialized.")

        del self.__dict__[_FROZEN]
        return output

    return init_method


def _get_wrapper(get_func):
    @ft.wraps(get_func)
    def get_method(self, key: str) -> Any:
        # avoid non-scalar error, raised by `jax` transformation
        # if a frozen value is returned.
        return unfreeze(get_func(self, key))

    return get_method


def _flatten(tree) -> tuple[Any, tuple[str, dict[str, Any]]]:
    """Flatten rule for `jax.tree_flatten`"""
    # in essence anything not declared in dataclass fields will be considered static
    static, dynamic = dict(vars(tree)), dict()
    for key in getattr(tree, _FIELD_MAP):
        dynamic[key] = static.pop(key)

    return dynamic.values(), (dynamic.keys(), static)


def _unflatten(cls, treedef, leaves):
    """Unflatten rule for `jax.tree_unflatten`"""
    tree = object.__new__(cls)  # do not call cls constructor
    tree.__dict__.update(treedef[1])
    tree.__dict__.update(zip(treedef[0], leaves))
    return tree


def _init_subclass(cls):
    # Non-decorated subclasses uses the base `treeclass` leaves only
    # this behavior is aligned with `dataclasses` not registering non-decorated
    # subclasses dataclass fields. for example:
    # >>> @treeclass
    # ... class A:
    # ...   a:int=1
    # >>> class B(A):
    # ...    b:int=2
    # >>> tree = B()
    # >>> jax.tree_leaves(tree)
    # [1]
    jtu.register_pytree_node(cls, _flatten, ft.partial(_unflatten, cls))
    return cls


def treeclass(cls):
    """Decorator to convert a class to a `treeclass`

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import pytreeclass as pytc

        >>> @pytc.treeclass
        ... class Linear :
        ...     weight : jnp.ndarray
        ...     bias   : jnp.ndarray

        >>> def __init__(self,key,in_dim,out_dim):
        ...    self.weight = jax.random.normal(key,shape=(in_dim, out_dim)) * jnp.sqrt(2/in_dim)
        ...    self.bias = jnp.ones((1,out_dim))

        >>> def __call__(self,x):
        ...    return x @ self.weight + self.bias

    Raises:
        TypeError: if the input class is not a `class`
    """
    if not isinstance(cls, type):
        # class decorator must be applied to a class
        raise TypeError(f"Expected `class` but got `{type(cls)}`.")

    for method_name in (
        "__setattr__",
        "__delattr__",
        "__getattribute__",
        "__getattr__",
    ):
        if method_name in vars(cls):
            # explicitly ensure that the class does not define these methods
            # even though it is being overriden by the decorator
            raise AttributeError(f"Cannot define `{method_name}` in {cls.__name__}.")

    FIELD_MAP = _generate_field_map(cls)

    attrs = dict()

    # dataclass containers
    attrs[_FIELD_MAP] = FIELD_MAP
    attrs[_FROZEN] = True

    # class constructor
    attrs["__init__"] = _init_wrapper(_retrieve_init_method(cls, FIELD_MAP))
    attrs["__new__"] = _new_wrapper(getattr(cls, "__new__"))
    attrs["__init_subclass__"] = _init_subclass

    # immutable methods
    attrs["__getattribute__"] = _get_wrapper(getattr(cls, "__getattribute__"))
    attrs["__delattr__"] = _delattr
    attrs["__setattr__"] = _setattr

    bases = (cls, _TreePretty, _TreeOperator, _TreeIndexer)
    cls = type(cls.__name__, bases, attrs)

    try:
        # in case of the base class is being initialized
        # then register all subclasses
        # however, do not consider the non-decorated subclasses leaves
        # for example:
        # >>> @treeclass
        # ... class A:
        # ...   a:int=1
        # >>> class B(A):
        # ...    b:int=2
        # >>> tree = B()
        # >>> jax.tree_leaves(tree)
        # [1]
        jtu.register_pytree_node(cls, _flatten, ft.partial(_unflatten, cls))
    except ValueError:
        # ignore if the class is already registered
        # this happens in cases of decorating by `treeclass`
        # and inheriting from `treeclass` decorated class
        # so the registeration is done twice, initially by the base class
        # and then by then by the decorator
        # for example:
        # >>> @treeclass
        # ... class A:
        # ...   a:int=1

        # >>> @treeclass
        # ... class B(A):
        # ...    b:int=2
        # >>> tree = B()
        # >>> jax.tree_leaves(tree)
        # [1, 2]
        pass

    return cls


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
