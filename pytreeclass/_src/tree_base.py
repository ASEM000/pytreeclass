from __future__ import annotations

import functools as ft
from typing import Any

import jax.tree_util as jtu
import numpy as np

from pytreeclass._src.tree_decorator import (
    _FIELD_MAP,
    _FROZEN,
    _MISSING,
    Field,
    ImmutableTreeError,
    _patch_init_method,
)
from pytreeclass._src.tree_freeze import FrozenWrapper
from pytreeclass._src.tree_indexer import _TreeAtIndexer
from pytreeclass._src.tree_operator import _TreeOperator
from pytreeclass.tree_viz.tree_pprint import _TreePretty

PyTree = Any

_POST = "__post_init__"


def _setattr(tree: PyTree, key: str, value: Any) -> None:
    """Set the attribute of the tree if the tree is not frozen"""
    if getattr(tree, _FROZEN):
        msg = f"Cannot set {key}={value!r}. Use `.at['{key}'].set({value!r})` instead."
        raise ImmutableTreeError(msg)

    object.__setattr__(tree, key, value)

    if hasattr(value, _FIELD_MAP) and (key not in getattr(tree, _FIELD_MAP)):
        field = Field(name=key, type=tuple)
        # register it to dataclass fields
        getattr(tree, _FIELD_MAP)[key] = field


def _delattr(tree, key: str) -> None:
    """Delete the attribute of the  if tree is not frozen"""
    if getattr(tree, _FROZEN):
        raise ImmutableTreeError(f"Cannot delete {key}.")
    object.__delattr__(tree, key)


def _new_wrapper(new_func):
    @ft.wraps(new_func)
    def new_method(cls, *a, **k) -> PyTree:
        self = object.__new__(cls)
        field_map = getattr(cls, _FIELD_MAP)
        # shadow the field map class attribute with an instance attribute
        object.__setattr__(self, _FIELD_MAP, field_map)

        for key in field_map:
            if field_map[key].default is not _MISSING:
                object.__setattr__(self, key, field_map[key].default)
            elif field_map[key].default_factory is not _MISSING:
                # call the default factory to get the default value
                object.__setattr__(self, key, field_map[key].default_factory())
        return self

    return new_method


def _init_wrapper(init_func):
    @ft.wraps(init_func)
    def init_method(self, *a, **k) -> None:
        object.__setattr__(self, _FROZEN, False)
        output = init_func(self, *a, **k)
        # in case __post_init__ is defined then call it
        # after the tree is initialized
        # here, we assume that __post_init__ is a method
        if _POST in vars(self.__class__):
            output = getattr(self, _POST)()
        object.__setattr__(self, _FROZEN, True)
        return output

    return init_method


def _flatten(tree) -> tuple[Any, tuple[str, dict[str, Any]]]:
    """Flatten rule for `jax.tree_flatten`"""
    # in essence anything not declared as a dataclass fields will be considered static
    static, dynamic = dict(tree.__dict__), dict()
    # avoid mutating the original dict by making a copy of dataclass fields
    static[_FIELD_MAP] = dict(static[_FIELD_MAP])

    for field in static[_FIELD_MAP].values():
        if isinstance(field, FrozenWrapper):
            # expose static fields as static leaves (FrozenWrapper)
            static[_FIELD_MAP][field.name] = (field).unwrap()
            dynamic[field.name] = FrozenWrapper(static.pop(field.name))
            continue

        # normal fields as dynamic leaves
        dynamic[field.name] = static.pop(field.name)
        continue

    return dynamic.values(), (dynamic.keys(), static)


def _unflatten(cls, treedef, leaves):
    """Unflatten rule for `jax.tree_unflatten`"""
    tree = object.__new__(cls)  # do not call cls constructor
    static = treedef[1]
    dynamic = dict(zip(treedef[0], leaves))

    for key in dynamic:
        if isinstance(dynamic[key], FrozenWrapper):
            # convert frozen value (static leaf) -> frozen field (to metadata)
            dynamic[key] = (dynamic[key]).unwrap()
            static[_FIELD_MAP][key] = FrozenWrapper(static[_FIELD_MAP][key])

    tree.__dict__.update(static)
    tree.__dict__.update(dynamic)
    return tree


def treeclass(cls=None, *, order: bool = True, repr: bool = True):
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
        repr: if `True` the `treeclass` will have a `__repr__`/ `__str__` method (default: `True`)
        frozen: if `True` the `treeclass` will be immutable (default: `True`) - use with caution -

    Returns:`
        `treeclass` of the input class

    Raises:
        TypeError: if the input class is not a `class`
    """

    def decorator(cls, order, repr):
        # generate and register field map to class.
        # generate init method if not defined based of the fields map
        # this is similar to dataclass decorator but without the bloated functionality
        cls = _patch_init_method(cls)

        attrs = dict(cls.__dict__)

        # initialize class
        attrs.update(__new__=_new_wrapper(cls.__new__))
        attrs.update(__init__=_init_wrapper(cls.__init__))

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

        # register the class to jax pytree
        return jtu.register_pytree_node_class(cls)

    if cls is None:
        return lambda cls: decorator(cls, order, repr)  # @treeclass
    return decorator(cls, order, repr)  # @treeclass(...)


def is_tree_equal(lhs: Any, rhs: Any) -> bool:
    """Assert if two pytrees are equal"""
    lhs_leaves, lhs_treedef = jtu.tree_flatten(lhs)
    rhs_leaves, rhs_treedef = jtu.tree_flatten(rhs)

    def is_equal(lhs_node, rhs_node):
        if hasattr(lhs_node, "dtype") and hasattr(rhs_node, "shape"):
            if hasattr(lhs_node, "dtype") and hasattr(rhs_node, "shape"):
                return np.all(lhs_node == rhs_node)
            return False
        return lhs_node == rhs_node

    if not (lhs_treedef == rhs_treedef):
        return False

    for (lhs, rhs) in zip(lhs_leaves, rhs_leaves):
        if not is_equal(lhs, rhs):
            return False
    return True
